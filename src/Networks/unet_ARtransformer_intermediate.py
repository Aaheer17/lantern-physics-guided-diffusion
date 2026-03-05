# created by Farzana with the help of Chatgpt and claude ai
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0):
        super().__init__()
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )
        
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]  # shape: (B, C, 1, 1)
        
        h = self.block2(h)
        
        return h + self.res_conv(x)
    


    
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        
        # QKV projection
        qkv = self.qkv(x_norm).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv.unbind(1)  # each shape: (b, heads, dim, seq)
        
        # Transpose to (b, heads, seq, dim)
        q, k, v = [t.permute(0, 1, 3, 2) for t in (q, k, v)]  # (b, heads, seq, dim)

        # Scaled dot-product attention
        attn = torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(q.shape[-1])
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)  # (b, heads, seq, dim)

        # Reshape back to (b, c, h, w)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return self.proj(out) + x

    
class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim, has_attention=False, dropout=0.0):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.res_block = ResidualBlock(in_channels + skip_channels, out_channels, time_emb_dim, dropout)
        self.attention = AttentionBlock(out_channels) if has_attention else nn.Identity()

    def forward(self, x, skip, time_emb):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x, time_emb)
        x = self.attention(x)
        return x
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, has_attention=False, dropout=0.0):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, time_emb_dim, dropout)
        self.attention = AttentionBlock(out_channels) if has_attention else nn.Identity()
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        x = self.attention(x)
        skip = x
        x = self.downsample(x)
        return x, skip
    
class ConditionProjection(nn.Module):
    def __init__(self, condition_dim, target_channels, spatial_size):
        super().__init__()
        self.spatial_size = spatial_size  # (H, W)
        self.target_channels = target_channels

        flat_size = target_channels * spatial_size[0] * spatial_size[1]
        self.proj = nn.Sequential(
            nn.Linear(condition_dim, flat_size),
            nn.SiLU(),
            nn.Linear(flat_size, flat_size),
        )

    def forward(self, condition):
        """
        condition: (B, condition_dim)
        returns: (B, target_channels, H, W)
        """
        b = condition.size(0)
        out = self.proj(condition)
        return out.view(b, self.target_channels, *self.spatial_size)

    
    
class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, condition_dim=None, 
                 model_channels=64, num_res_blocks=2, attention_resolutions=[8],
                 dropout=0.0, channel_mult=[1, 2, 4], time_embed_dim=None):
        super().__init__()
        
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        
        # Time embedding
        if time_embed_dim is None:
            time_embed_dim = model_channels * 4
        self.time_embed_dim = time_embed_dim
        
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Condition projection (if provided)
        self.condition_proj = None
        if condition_dim is not None:
            # Project condition to spatial feature map
            self.condition_proj = ConditionProjection(
                condition_dim, model_channels, (16, 9)  # Your spatial dimensions
            )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_channels = [model_channels]
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                has_attention = (16 // (2 ** level)) in attention_resolutions
                self.down_blocks.append(DownBlock(
                    ch, out_ch, time_embed_dim, has_attention, dropout
                ))
                ch = out_ch
                input_channels.append(ch)
        
        # Middle blocks
        self.middle_block1 = ResidualBlock(ch, ch, time_embed_dim, dropout)
        self.middle_attention = AttentionBlock(ch)
        self.middle_block2 = ResidualBlock(ch, ch, time_embed_dim, dropout)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for level, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                skip_ch = input_channels.pop()
                has_attention = (16 // (2 ** (len(channel_mult) - 1 - level))) in attention_resolutions
                self.up_blocks.append(UpBlock(
                    ch, out_ch, time_embed_dim, has_attention, dropout
                ))
                ch = out_ch
        
        # Final layers
        self.final_res_block = ResidualBlock(ch, ch, time_embed_dim, dropout)
        self.final_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x, time, condition=None):
        # x: [B, C, H, W] - spatial slice
        # time: [B] - time steps
        # condition: [B, condition_dim] - condition embedding
        
        # Time embedding
        time_emb = self.time_embed(time)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Add condition if provided
        if condition is not None and self.condition_proj is not None:
            condition_features = self.condition_proj(condition)
            x = x + condition_features
        
        # Store skip connections
        skips = [x]
        
        # Downsampling
        for down_block in self.down_blocks:
            x, skip = down_block(x, time_emb)
            skips.append(skip)
        
        # Middle blocks
        x = self.middle_block1(x, time_emb)
        x = self.middle_attention(x)
        x = self.middle_block2(x, time_emb)
        
        # Upsampling
        for up_block in self.up_blocks:
            skip = skips.pop()
            x = up_block(x, skip, time_emb)
        
        # Final layers
        x = self.final_res_block(x, time_emb)
        x = self.final_conv(x)
        
        return x


# Modified build_subnet function for your ARTransformer5D class
# def build_unet_subnet(self):
#     """Build UNet-based subnet for diffusion/flow matching"""
    
#     # Calculate condition dimension
#     cond_dim = self.encode_t_dim + self.dim_embedding
#     if self.layer_cond:
#         cond_dim += self.seq_len
    
#     # Create UNet with appropriate dimensions
#     unet = UNet2D(
#         in_channels=1,  # Single channel input
#         out_channels=1,  # Single channel output
#         condition_dim=cond_dim,
#         model_channels=self.params.get("unet_channels", 64),
#         num_res_blocks=self.params.get("unet_res_blocks", 2),
#         attention_resolutions=self.params.get("unet_attention_res", [8]),
#         dropout=self.params.get("dropout", 0.0),
#         channel_mult=self.params.get("unet_channel_mult", [1, 2, 4]),
#         time_embed_dim=self.params.get("unet_time_embed_dim", None)
#     )
    
#     return unet


# Wrapper class to handle the interface with your existing code
class UNetWrapper(nn.Module):
    def __init__(self, unet, spatial_h, spatial_w, encode_t_dim):
        super().__init__()
        self.unet = unet
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.encode_t_dim = encode_t_dim
        
    def forward(self, x):
        # x: [B, spatial_dim + time_dim + condition_dim]
        # This is the fallback MLP interface - shouldn't be used with UNet
        # but keeping for compatibility
        batch_size = x.shape[0]
        
        # Parse input
        spatial_dim = self.spatial_h * self.spatial_w
        spatial_data = x[:, :spatial_dim]  # [B, spatial_dim]
        time_and_condition = x[:, spatial_dim:]  # [B, time_dim + condition_dim]
        
        # Reshape spatial data to 2D
        spatial_2d = spatial_data.view(batch_size, 1, self.spatial_h, self.spatial_w)
        
        # Extract time and condition data
        time_data = time_and_condition[:, :self.encode_t_dim]
        condition_data = time_and_condition[:, self.encode_t_dim:]
        
        # For compatibility, create dummy time steps
        time_steps = torch.zeros(batch_size, device=x.device)
        
        # Forward through UNet
        output = self.unet(spatial_2d, time_steps, condition_data)
        
        # Flatten output back to spatial_dim
        return output.view(batch_size, spatial_dim)
    
# unet = build_unet_subnet()  # Assuming 'self' context or pass required args
# model = UNetWrapper(unet, spatial_h=16, spatial_w=9, encode_t_dim=32)

# x = torch.randn(4, 16*9 + 32 + 64)  # e.g., [B, spatial + time + cond]
# out = model(x)
# print(out.shape)  # should be [4, 144] 