import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for time conditioning."""
    
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
    """Residual block with time and condition embedding."""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim=0, 
                 dropout=0.0, activation="SiLU", normalization="GroupNorm"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Normalization layers
        if normalization == "GroupNorm":
            self.norm1 = nn.GroupNorm(8, in_channels)
            self.norm2 = nn.GroupNorm(8, out_channels)
        elif normalization == "BatchNorm":
            self.norm1 = nn.BatchNorm2d(in_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            
        # Activation
        self.activation = getattr(nn, activation)()
        
        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            self.activation,
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Condition embedding projection (if used)
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                self.activation,
                nn.Linear(cond_dim, out_channels)
            )
        else:
            self.cond_mlp = None
            
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, time_emb, cond_emb=None):
        residual = self.residual_conv(x)
        
        # First conv block
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        # Add condition embedding if available
        if cond_emb is not None and self.cond_mlp is not None:
            cond_emb = self.cond_mlp(cond_emb)
            h = h + cond_emb[:, :, None, None]
        
        # Second conv block
        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + residual


class AttentionBlock(nn.Module):
    """Self-attention block for U-Net."""
    
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        x = x.view(B, C, H * W)
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H * W)
        k = k.view(B, self.num_heads, self.head_dim, H * W)
        v = v.view(B, self.num_heads, self.head_dim, H * W)
        
        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bhij,bhdj->bhdi', attn, v)
        out = out.contiguous().view(B, C, H * W)
        
        out = self.proj_out(out)
        out = out.view(B, C, H, W)
        
        return out + residual


class UNetDiffusion(nn.Module):
    """U-Net architecture for diffusion reverse network."""
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        # Extract parameters with defaults
        self.in_channels = params.get("in_channels", 3)
        self.out_channels = params.get("out_channels", 3)
        self.base_channels = params.get("base_channels", 128)
        self.channel_mults = params.get("channel_mults", [1, 2, 4, 8])
        self.num_res_blocks = params.get("num_res_blocks", 2)
        self.dropout = params.get("dropout", 0.0)
        self.activation = params.get("activation", "SiLU")
        self.normalization = params.get("normalization", "GroupNorm")
        self.use_attention = params.get("use_attention", True)
        self.attention_resolutions = params.get("attention_resolutions", [16, 8])
        
        # Time embedding
        self.time_dim = params.get("time_dim", 256)
        self.time_embedding = SinusoidalPositionEmbedding(self.time_dim // 4)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim // 4, self.time_dim),
            getattr(nn, self.activation)(),
            nn.Linear(self.time_dim, self.time_dim)
        )
        
        # Condition embedding (if used)
        self.cond_dim = params.get("cond_dim", 0)
        if self.cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(self.cond_dim, self.time_dim),
                getattr(nn, self.activation)(),
                nn.Linear(self.time_dim, self.time_dim)
            )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(self.in_channels, self.base_channels, 3, padding=1)
        
        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        ch = self.base_channels
        for level, mult in enumerate(self.channel_mults):
            out_ch = self.base_channels * mult
            
            # Residual blocks for this level
            for _ in range(self.num_res_blocks):
                block = ResidualBlock(
                    ch, out_ch, self.time_dim, self.cond_dim,
                    dropout=self.dropout,
                    activation=self.activation,
                    normalization=self.normalization
                )
                self.down_blocks.append(block)
                ch = out_ch
                
                # Add attention if specified
                if self.use_attention and any(ch == self.base_channels * m for m in self.attention_resolutions):
                    self.down_blocks.append(AttentionBlock(ch))
            
            # Downsample (except for the last level)
            if level < len(self.channel_mults) - 1:
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            else:
                self.down_samples.append(nn.Identity())
        
        # Middle blocks
        mid_ch = self.base_channels * self.channel_mults[-1]
        self.mid_block1 = ResidualBlock(
            mid_ch, mid_ch, self.time_dim, self.cond_dim,
            dropout=self.dropout,
            activation=self.activation,
            normalization=self.normalization
        )
        if self.use_attention:
            self.mid_attn = AttentionBlock(mid_ch)
        self.mid_block2 = ResidualBlock(
            mid_ch, mid_ch, self.time_dim, self.cond_dim,
            dropout=self.dropout,
            activation=self.activation,
            normalization=self.normalization
        )
        
        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level, mult in enumerate(reversed(self.channel_mults)):
            out_ch = self.base_channels * mult
            
            for i in range(self.num_res_blocks + 1):
                # Skip connection from encoder
                skip_ch = self.base_channels * mult if i == 0 else 0
                if level == 0 and i == 0:
                    skip_ch = self.base_channels * self.channel_mults[-2] if len(self.channel_mults) > 1 else self.base_channels
                
                block = ResidualBlock(
                    ch + skip_ch, out_ch, self.time_dim, self.cond_dim,
                    dropout=self.dropout,
                    activation=self.activation,
                    normalization=self.normalization
                )
                self.up_blocks.append(block)
                ch = out_ch
                
                # Add attention if specified
                if self.use_attention and any(ch == self.base_channels * m for m in self.attention_resolutions):
                    self.up_blocks.append(AttentionBlock(ch))
            
            # Upsample (except for the last level)
            if level < len(self.channel_mults) - 1:
                self.up_samples.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            else:
                self.up_samples.append(nn.Identity())
        
        # Final output
        self.norm_out = nn.GroupNorm(8, self.base_channels) if self.normalization == "GroupNorm" else nn.Identity()
        self.conv_out = nn.Conv2d(self.base_channels, self.out_channels, 3, padding=1)
        
    def forward(self, x, time, cond=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            time: Time steps [B]
            cond: Conditional input [B, cond_dim] (optional)
        """
        # Time embedding
        time_emb = self.time_embedding(time)
        time_emb = self.time_mlp(time_emb)
        
        # Condition embedding
        cond_emb = None
        if cond is not None and hasattr(self, 'cond_mlp'):
            cond_emb = self.cond_mlp(cond)
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Store skip connections
        skip_connections = [h]
        
        # Encoder
        block_idx = 0
        for level in range(len(self.channel_mults)):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[block_idx](h, time_emb, cond_emb)
                block_idx += 1
                
                # Apply attention if it exists
                if block_idx < len(self.down_blocks) and isinstance(self.down_blocks[block_idx], AttentionBlock):
                    h = self.down_blocks[block_idx](h)
                    block_idx += 1
                
                skip_connections.append(h)
            
            # Downsample
            h = self.down_samples[level](h)
            if level < len(self.channel_mults) - 1:
                skip_connections.append(h)
        
        # Middle
        h = self.mid_block1(h, time_emb, cond_emb)
        if hasattr(self, 'mid_attn'):
            h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb, cond_emb)
        
        # Decoder
        block_idx = 0
        for level in range(len(self.channel_mults)):
            for i in range(self.num_res_blocks + 1):
                # Skip connection
                if skip_connections:
                    skip = skip_connections.pop()
                    h = torch.cat([h, skip], dim=1)
                
                h = self.up_blocks[block_idx](h, time_emb, cond_emb)
                block_idx += 1
                
                # Apply attention if it exists
                if block_idx < len(self.up_blocks) and isinstance(self.up_blocks[block_idx], AttentionBlock):
                    h = self.up_blocks[block_idx](h)
                    block_idx += 1
            
            # Upsample
            h = self.up_samples[level](h)
        
        # Final output
        h = self.norm_out(h)
        h = getattr(nn, self.activation)()(h)
        h = self.conv_out(h)
        
        return h


# Example usage and parameter configuration
def create_unet_diffusion_model():
    """Create a U-Net diffusion model with example parameters."""
    
    params = {
        "in_channels": 3,           # RGB images
        "out_channels": 3,          # RGB output
        "base_channels": 128,       # Base number of channels
        "channel_mults": [1, 2, 4, 8],  # Channel multipliers for each level
        "num_res_blocks": 2,        # Number of residual blocks per level
        "dropout": 0.1,             # Dropout rate
        "activation": "SiLU",       # Activation function
        "normalization": "GroupNorm", # Normalization type
        "use_attention": True,      # Use attention blocks
        "attention_resolutions": [4, 2],  # Resolutions to apply attention
        "time_dim": 256,            # Time embedding dimension
        "cond_dim": 128,            # Conditional embedding dimension (if used)
    }
    
    return UNetDiffusion(params)


# Test the model
if __name__ == "__main__":
    model = create_unet_diffusion_model()
    
    # Test forward pass
    batch_size = 2
    height, width = 64, 64
    
    x = torch.randn(batch_size, 3, height, width)
    time = torch.randint(0, 1000, (batch_size,))
    cond = torch.randn(batch_size, 128)  # Optional conditioning
    
    with torch.no_grad():
        output = model(x, time, cond)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")