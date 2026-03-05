import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange
from torchdiffeq import odeint
from typing import Optional
import inspect
import time
from .unet_ARTransformer import UNet2D, UNetWrapper

class CrossAttention(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
    
    def forward(self, x, cond):
        # x: [B, L_q, D], cond: [B, L_kv, D]
        # returns [B, L_q, D]
        return self.attn(x, cond, cond)[0]


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, activation="SiLU", dropout=0.0, normalization=None):
        super().__init__()
        layers = []

        layers.append(nn.Linear(dim, dim))
        if normalization:
            layers.append(getattr(nn, normalization)(dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(getattr(nn, activation)())
        
        layers.append(nn.Linear(dim, dim))  # Second layer in residual block

        self.block = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.block(x))


class ARTransformer5D(nn.Module):
    def __init__(self, params):
        super().__init__()
        # Read in the network specifications from the params
        self.params = params
        self.dim_embedding = self.params["dim_embedding"]
        
        # 5D data dimensions: B, 1, 45, 16, 9
        self.seq_len = 45  # autoregressive dimension
        self.spatial_h = 16  # spatial dimension 1
        self.spatial_w = 9   # spatial dimension 2
        self.spatial_dim = self.spatial_h * self.spatial_w  # flattened spatial dimension
        self.spatial_size = (16, 9)
        # Condition dimension: B, 46
        self.dims_c = 46
        
        self.bayesian = False
        self.layer_cond = self.params.get("layer_cond", False)
        self.gen_model = self.params['gen_model']
        self.inference_steps = self.params.get('inference_steps', 500)
        
        self.c_embed = self.params.get("c_embed", None)
        self.x_embed = self.params.get("x_embed", None)
        
        self.encode_t_dim = self.params.get("encode_t_dim", 64)
        self.encode_t_scale = self.params.get("encode_t_scale", 30)
        
        # Transformer for autoregressive modeling
        self.transformer = nn.Transformer(
            d_model=self.dim_embedding,
            nhead=params["n_head"],
            num_encoder_layers=params["n_encoder_layers"],
            num_decoder_layers=params["n_decoder_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params.get("dropout_transformer", 0.0),
            batch_first=True,
        )
        
        # Embedding networks
        if self.x_embed:
            # For spatial data (16*9 = 144 dimensions)
            self.x_embed = nn.Sequential(
                nn.Linear(self.spatial_dim, self.dim_embedding),
                nn.ReLU(),
                nn.Linear(self.dim_embedding, self.dim_embedding)
            )
        
        if self.c_embed:
            # For condition data (46 dimensions)
            self.c_embed = nn.Sequential(
                nn.Linear(self.dims_c, self.dim_embedding),
                nn.ReLU(),
                nn.Linear(self.dim_embedding, self.dim_embedding)
            )
        
        # Time embedding
        if self.gen_model == "DDPM":
            max_timesteps = self.params.get("num_timesteps", 1000)
            self.t_embed_discrete = nn.Embedding(max_timesteps, self.encode_t_dim)
            
        self.t_embed_continuous = nn.Sequential(
            GaussianFourierProjection(embed_dim=self.encode_t_dim, scale=self.encode_t_scale),
            nn.Linear(self.encode_t_dim, self.encode_t_dim)
        )
        
        self.t_embed = self.t_embed_continuous
        
        # Subnet for diffusion/flow matching
        #self.subnet = self.build_subnet()
        self.subnet=self.build_unet_subnet(
        encode_t_dim=self.encode_t_dim,
        dim_embedding=self.dim_embedding,
        seq_len=self.seq_len,
        layer_cond=self.layer_cond,
        spatial_size=self.spatial_size
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=self.dim_embedding, 
            max_len=max(self.seq_len, self.dims_c) + 1, 
            dropout=0.0
        )
        
        self.num_timesteps = self.params.get("num_timesteps", 1000)

    def compute_embedding_5d(self, data, dim, embedding_net, is_condition=False):
        """
        Compute embeddings for 5D data with positional encoding
        
        Args:
            data: tensor of shape [B, seq_len, spatial_h, spatial_w] or [B, condition_dim]
            dim: dimension for one-hot encoding
            embedding_net: embedding network
            is_condition: whether this is condition data
        """
        batch_size = data.shape[0]
        device = data.device
        dtype = data.dtype
        
        if is_condition:
            # Condition data: [B, 46] -> [B, 1, 46] -> [B, 1, dim_embedding]
            if data.ndim == 2:
                data = data.unsqueeze(1)  # [B, 1, 46]
            
            if embedding_net is None:
                # Zero-pad to reach dim_embedding
                n_rest = self.dim_embedding - data.shape[-1]
                assert n_rest >= 0
                zeros = torch.zeros((*data.shape[:2], n_rest), device=device, dtype=dtype)
                return torch.cat((data, zeros), dim=2)
            else:
                # Use embedding network
                embedded = embedding_net(data)  # [B, 1, dim_embedding]
                return self.positional_encoding(embedded)
        else:
            # Data: [B, seq_len, spatial_h, spatial_w] -> [B, seq_len, spatial_dim]
            #print("data.ndim: ",data.ndim)
            if data.ndim == 5:
                # Convert [B, C, seq_len, H, W] -> [B, seq_len, C*H*W]
                b, c, s, h, w = data.shape
                data = data.permute(0, 2, 1, 3, 4).contiguous()  # [B, seq_len, C, H, W]
                data = data.view(b, s, c * h * w)  # [B, seq_len, spatial_dim]
            elif data.ndim == 4:
                data = data.view(batch_size, -1, self.spatial_dim)
            #print("after changes of shape in compute embedding ",data.shape)
            # Add positional information for sequence
            seq_len = data.shape[1]
            one_hot = torch.eye(seq_len, device=device, dtype=dtype)[None, :, :].expand(batch_size, -1, -1)
            
            if embedding_net is None:
                # Zero-pad to reach dim_embedding
                n_rest = self.dim_embedding - data.shape[-1] - seq_len
                assert n_rest >= 0
                zeros = torch.zeros((*data.shape[:2], n_rest), device=device, dtype=dtype)
                return torch.cat((data, one_hot, zeros), dim=2)
            else:
                # Use embedding network
                embedded = embedding_net(data)  # [B, seq_len, dim_embedding]
                return self.positional_encoding(embedded)

    def build_subnet(self):
        """Build the subnet for diffusion/flow matching"""
        self.intermediate_dim = self.params.get("intermediate_dim", 512)
        self.dropout = self.params.get("dropout", 0.0)
        self.activation = self.params.get("activation", "SiLU")
        self.layers_per_block = self.params.get("layers_per_block", 8)
        self.normalization = self.params.get("normalization", None)
        self.residual = self.params.get('residual', None)

        # Input dimension: spatial_dim + time_dim + condition_dim
        cond_dim = self.encode_t_dim + self.dim_embedding
        if self.layer_cond:
            cond_dim += self.seq_len
        #print("shape of cond_dim: ",cond_dim)

        linear = nn.Linear(self.spatial_dim + cond_dim, self.intermediate_dim)
        #linear = nn.Linear(6608, self.intermediate_dim)
        layers = [linear, getattr(nn, self.activation)()]

        for _ in range(1, self.layers_per_block - 1):
            if self.residual:
                layers.append(ResidualMLPBlock(self.intermediate_dim, activation=self.activation, 
                                       dropout=self.dropout, normalization=self.normalization))
            else:
                linear = nn.Linear(self.intermediate_dim, self.intermediate_dim)
                layers.append(linear)
                if self.normalization is not None:
                    layers.append(getattr(nn, self.normalization)(self.intermediate_dim))
                if self.dropout is not None:
                    layers.append(nn.Dropout(p=self.dropout))
                layers.append(getattr(nn, self.activation)())

        # Output spatial dimensions
        linear = nn.Linear(self.intermediate_dim, self.spatial_dim)
        layers.append(linear)

        return nn.Sequential(*layers)
    def build_unet_subnet(self, encode_t_dim, dim_embedding, seq_len=None, layer_cond=False, 
                     spatial_size=(16, 9), params=None):
        """Build UNet-based subnet for diffusion/flow matching"""

        if params is None:
            params = {}

        # Calculate condition dimension
        #cond_dim = encode_t_dim + dim_embedding
        cond_dim=dim_embedding
        if layer_cond and seq_len is not None:
            cond_dim += seq_len

        #print(f"Building UNet with condition_dim={cond_dim}, spatial_size={spatial_size}")

        # Create UNet with appropriate dimensions
        unet = UNet2D(
            in_channels=1,
            out_channels=1,
            condition_dim=cond_dim,
            model_channels=params.get("unet_channels", 64),
            num_res_blocks=params.get("unet_res_blocks", 2),
            attention_resolutions=params.get("unet_attention_res", [8]),
            dropout=params.get("dropout", 0.0),
            channel_mult=params.get("unet_channel_mult", [1, 2, 4]),
            time_embed_dim=params.get("unet_time_embed_dim", None),
            spatial_size=spatial_size
        )

        return UNetWrapper(unet, spatial_size[0], spatial_size[1], self.encode_t_dim, cond_dim)


    def get_time_embedding(self, t):
        """Get time embedding based on model type"""
        if self.gen_model == "DDPM":
            if t.dtype == torch.long:
                return self.t_embed_discrete(t)
            else:
                return self.t_embed_discrete(t.long())
        else:
            return self.t_embed_continuous(t.float())

    def sample_spatial_slice(self, c_embed, scheduler=None, sampling_type='ddpm'):
        """
        Sample one spatial slice (16x9) given condition embedding
        """
        batch_size = c_embed.size(0)
        device = c_embed.device
        dtype = c_embed.dtype

        if sampling_type == "cfm":
            # CFM sampling with ODE
            x_0 = torch.randn((batch_size, self.spatial_dim), device=device, dtype=dtype)
            
            def net_wrapper(t, x_t):
                t_torch = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)
                t_torch = self.t_embed_continuous(t_torch)
                v = self.subnet(torch.cat([x_t, t_torch.view(batch_size, -1), c_embed.view(batch_size, -1)], dim=-1))
                return v

            with torch.inference_mode():
                x_t = odeint(
                    net_wrapper, x_0, torch.tensor([0, 1], dtype=dtype, device=device),
                    **self.params.get("solver_kwargs", {})
                )
            return x_t[-1].view(batch_size, 1, self.spatial_h, self.spatial_w)
            
        elif sampling_type == "ddim":
            # DDIM sampling
            x_t = torch.randn((batch_size, self.spatial_dim), device=device, dtype=dtype)
            step_indices = torch.linspace(0, self.num_timesteps - 1, self.inference_steps, dtype=torch.long).flip(0).to(device)
            
            for i in range(len(step_indices) - 1):
                t = step_indices[i]
                t_prev = step_indices[i + 1]
                t_batch = t.expand(batch_size)
                t_prev_batch = t_prev.expand(batch_size)

                eps_pred = self.subnet(torch.cat([
                    x_t,
                    self.get_time_embedding(t_batch).view(batch_size, -1),
                    c_embed.view(batch_size, -1)
                ], dim=-1))

                alpha_t = scheduler['sqrt_alphas_cumprod'].to(x_t.device)[t_batch].view(-1, 1)
                alpha_prev = scheduler['sqrt_alphas_cumprod'].to(x_t.device)[t_prev_batch].view(-1, 1)
                sqrt_one_minus_alpha_t = scheduler['sqrt_one_minus_alphas_cumprod'].to(x_t.device)[t_batch].view(-1, 1)

                x_0_pred = (x_t - sqrt_one_minus_alpha_t * eps_pred) / alpha_t
                dir_xt = torch.sqrt(1.0 - alpha_prev ** 2) * eps_pred
                x_t = alpha_prev * x_0_pred + dir_xt

            # Final step
            t = step_indices[-1].expand(batch_size)
            eps_pred = self.subnet(torch.cat([
                x_t,
                self.get_time_embedding(t).view(batch_size, -1),
                c_embed.view(batch_size, -1)
            ], dim=-1))
            
            alpha_t = scheduler['sqrt_alphas_cumprod'].to(x_t.device)[t].view(-1, 1)
            sqrt_one_minus_alpha_t = scheduler['sqrt_one_minus_alphas_cumprod'].to(x_t.device)[t].view(-1, 1)
            x_0_pred = (x_t - sqrt_one_minus_alpha_t * eps_pred) / alpha_t
            
            return x_0_pred.view(batch_size, 1, self.spatial_h, self.spatial_w)
        
        else:  # DDPM
            x_t = torch.randn((batch_size, self.spatial_dim), device=device, dtype=dtype)
            
            for i in reversed(range(self.inference_steps)):
                t = torch.full((batch_size,), i, dtype=torch.long, device=device)
                
                eps_pred = self.subnet(torch.cat([
                    x_t,
                    self.get_time_embedding(t).view(batch_size, -1),
                    c_embed.view(batch_size, -1)
                ], dim=-1))
                
                # Predict x_0 and get posterior
                alpha_t = scheduler['sqrt_alphas_cumprod'].to(x_t.device)[t].view(-1, 1)
                sqrt_one_minus_alpha_t = scheduler['sqrt_one_minus_alphas_cumprod'].to(x_t.device)[t].view(-1, 1)
                x_0_pred = (x_t - sqrt_one_minus_alpha_t * eps_pred) / alpha_t
                
                # Compute posterior mean and variance
                coef1 = scheduler["posterior_mean_coef1"].to(x_t.device)[t].view(-1, 1)
                coef2 = scheduler["posterior_mean_coef2"].to(x_t.device)[t].view(-1, 1)
                var = scheduler["posterior_variance"].to(x_t.device)[t].view(-1, 1)
                
                mu = coef1 * x_0_pred + coef2 * x_t
                
                if i > 0:
                    noise = torch.randn_like(x_t)
                    x_t = mu + torch.sqrt(var) * noise
                else:
                    x_t = mu
            
            return x_t.view(batch_size, 1, self.spatial_h, self.spatial_w)

    def forward(self, c, x_t=None, t=None, x=None, rev=False, diffusion_buffers=None, sampling_type='ddpm'):
        """
        Forward pass for 5D autoregressive transformer
        
        Args:
            c: condition tensor [B, 46]
            x_t: noisy data [B, seq_len, spatial_h, spatial_w] during training
            t: time steps
            x: clean data [B, seq_len, spatial_h, spatial_w] during training
            rev: reverse mode (sampling)
            diffusion_buffers: scheduler buffers for DDPM/DDIM
            sampling_type: 'cfm', 'ddpm', 'ddim', or 'plms'
        """
        
        if not rev:
            # Training mode - autoregressive masking
            # print(f'Forward pass for {self.gen_model} training')
            # print(f'Input shapes - c: {c.shape}, x: {x.shape}, x_t: {x_t.shape}')
            
            # Autoregressive padding: [B, seq_len, H, W] -> [B, seq_len, H, W]
            # Pad with zeros at the beginning of sequence
            #xp = F.pad(x[:, :-1], (0, 0, 0, 0, 1, 0))  # Pad sequence dimension
            xp = F.pad(x[:, :, :-1], (0, 0, 0, 0, 1, 0))  # Pad along the layer (sequence) dimension

            # print("shape of xp: ",xp.shape)
            # Create embeddings
            src_transformer = self.compute_embedding_5d(c, dim=self.dims_c, embedding_net=self.c_embed, is_condition=True)
            tgt_transformer = self.compute_embedding_5d(xp, dim=self.seq_len, embedding_net=self.x_embed, is_condition=False)
            
            # Causal mask for autoregressive modeling
            #xformer_tgt_mask = torch.ones((xp.shape[1], xp.shape[1]), device=x.device, dtype=torch.bool).triu(diagonal=1)
            seq_len = xp.shape[2]
            xformer_tgt_mask = torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool).triu(diagonal=1)
            
            #print(f'Transformer input shapes - src: {src_transformer.shape}, tgt: {tgt_transformer.shape}')
            
            # Transformer forward pass
            embedding = self.transformer(
                src=src_transformer,
                tgt=tgt_transformer,
                tgt_mask=xformer_tgt_mask,
            )
            
            #print(f'Transformer output shape: {embedding.shape}')
            
            # Layer conditioning if enabled
            if self.layer_cond:
                layer_one_hot = repeat(
                    torch.eye(self.seq_len, device=x.device), '... -> b ...', b=len(c)
                )
                embedding = torch.cat([embedding, layer_one_hot], dim=2)
            
            # Time embedding
            t_embedded = self.get_time_embedding(t)
            #t_embedded = t_embedded.unsqueeze(1).expand(-1, x_t.size(1), -1)
            #t_embedded = t_embedded.expand(-1, x_t.size(1), -1)
            
            # x_t: [B, 1, 45, 16, 9]
            x_t = x_t.permute(0, 2, 1, 3, 4).contiguous()  # → [B, 45, 1, 16, 9]
            x_t_flat = x_t.view(x_t.size(0), x_t.size(1), -1)  # → [B, 45, 144]
            # print("shape of x_t_flat: ", x_t_flat.shape)
            
            t_embedded = t_embedded.expand(-1, x_t_flat.size(1), -1)  # ✅ expands to 45
            # VECTORIZED APPROACH: Process all slices at once
            # Reshape to treat each slice as a separate batch item
            batch_size, seq_len, spatial_dim = x_t_flat.shape

            # Flatten batch and sequence dimensions
            x_t_reshaped = x_t_flat.reshape(batch_size * seq_len, spatial_dim)  # [B*45, 144]
            t_reshaped = t_embedded.reshape(batch_size * seq_len, -1)  # [B*45, encode_t_dim]
            embedding_reshaped = embedding.reshape(batch_size * seq_len, -1)  # [B*45, dim_embedding]

            # Concatenate all inputs
            subnet_input = torch.cat([x_t_reshaped, t_reshaped, embedding_reshaped], dim=-1)
            #print("shape of subnet_input: ",subnet_input.shape)

            # Single forward pass through subnet for ALL slices
            pred_flat = self.subnet(subnet_input)  # [B*45, spatial_dim]

            # Reshape back to original structure
            pred = pred_flat.view(batch_size, seq_len, spatial_dim)  # [B, 45, 144]
            pred = pred.view(batch_size, seq_len, self.spatial_h, self.spatial_w)  # [B, 45, 16, 9]

            return pred
            # print("shape of t_embedded: ",t_embedded.shape)
              # done in loop for training
#             # Predict for each spatial slice
#             pred_list = []
#             loop_times=[]
#             before_start=time.time()
#             for i in range(x_t_flat.size(1)):  # i ∈ [0, 44]
                
#                 loop_start=time.time()
#                 # print(f"Loop index i: {i}")
#                 # print(f"x_t_flat[:, {i}:{i+1}, :].shape: {x_t_flat[:, i:i+1, :].shape}")
#                 # print(f"t_embedded[:, {i}:{i+1}, :].shape: {t_embedded[:, i:i+1, :].shape}")
#                 # print(f"embedding[:, {i}:{i+1}, :].shape: {embedding[:, i:i+1, :].shape}")
#                 slice_input = torch.cat([
#                     x_t_flat[:, i:i+1, :],        # [B, 1, spatial_dim]
#                     t_embedded[:, i:i+1, :],      # [B, 1, encode_t_dim]
#                     embedding[:, i:i+1, :]        # [B, 1, dim_embedding]
#                 ], dim=-1)

#                 #print("shape of slice_input: ", slice_input.shape)
#                 pred_slice = self.subnet(slice_input.squeeze(1))  # [B, spatial_dim]
#                 # print("shape of pred_slice: ", pred_slice.shape)
#                 pred_list.append(pred_slice.unsqueeze(1))
#                 loop_times.append(time.time() - loop_start)
                
#             print(f"Avg loop time: {sum(loop_times)/len(loop_times):.4f}s")
#             pred = torch.cat(pred_list, dim=1)  # [B, seq_len, spatial_dim]
#             pred = pred.view(x_t.size(0), x_t.size(1), self.spatial_h, self.spatial_w)
#             before_end=time.time()
#             print(f'Prediction shape: {pred.shape} and total prediction time: {before_end-before_start}')

            
           
            
        else:
            # Sampling mode - autoregressive generation
            # print(f"Sampling mode ({self.gen_model})")
            # print(f'Condition shape: {c.shape}')
            
            batch_size = c.shape[0]
            device = c.device
            dtype = c.dtype
            
            # Initialize with zero slice
            x = torch.zeros((batch_size, 1, self.spatial_h, self.spatial_w), device=device, dtype=dtype)
            
            # Condition embedding
            c_embed = self.compute_embedding_5d(c, dim=self.dims_c, embedding_net=self.c_embed, is_condition=True)
            
            # Autoregressive generation
            for i in range(self.seq_len):
                #print(f"Generating slice {i+1}/{self.seq_len}")
                
                # Compute transformer embedding for current sequence
                embedding = self.transformer(
                    src=c_embed,
                    tgt=self.compute_embedding_5d(x, dim=self.seq_len, embedding_net=self.x_embed, is_condition=False),
                    tgt_mask=torch.ones(
                        (x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool
                    ).triu(diagonal=1),
                )
                
                # Layer conditioning if enabled
                if self.layer_cond:
                    layer_one_hot = repeat(
                        F.one_hot(torch.tensor(i, device=x.device), self.seq_len),
                        'd -> b 1 d', b=batch_size
                    )
                    current_embedding = torch.cat([embedding[:, -1:, :], layer_one_hot], dim=2)
                else:
                    current_embedding = embedding[:, -1:, :]
                
                # Sample new spatial slice
                x_new = self.sample_spatial_slice(current_embedding, diffusion_buffers, sampling_type)
                
                # Append to sequence
                x = torch.cat((x, x_new), dim=1)
                #print(f"Current sequence shape: {x.shape}")
            
            # Remove initial zero slice
            pred = x[:, 1:]  # [B, seq_len, H, W]
            #print(f"Final prediction shape: {pred.shape}")
            
        return pred


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        if x.ndim == 1:
            x = x[:, None]
        x_proj = x * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)