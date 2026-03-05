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
                 dropout=0.0, channel_mult=[1, 2, 4], time_embed_dim=None,
                 spatial_size=(16, 9)):
        super().__init__()
        
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.spatial_size = spatial_size
        
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
            self.condition_proj = ConditionProjection(
                condition_dim, model_channels, spatial_size
            )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Build network structure
        self._build_down_blocks()
        self._build_middle_blocks()
        self._build_up_blocks()
        
        # Final layers
        self.final_res_block = ResidualBlock(model_channels, model_channels, time_embed_dim, dropout)
        self.final_conv = nn.Conv2d(model_channels, out_channels, 3, padding=1)
    
    def _build_down_blocks(self):
        """Build downsampling blocks"""
        self.down_blocks = nn.ModuleList()
        ch = self.model_channels
        self.down_channels = [ch]  # Track channels for skip connections
        
        for level, mult in enumerate(self.channel_mult):
            out_ch = self.model_channels * mult
            
            for _ in range(self.num_res_blocks):
                # Calculate current resolution for attention
                current_res = self.spatial_size[0] // (2 ** level)
                has_attention = current_res in self.attention_resolutions
                
                self.down_blocks.append(DownBlock(
                    ch, out_ch, self.time_embed_dim, has_attention, 0.0
                ))
                ch = out_ch
                self.down_channels.append(ch)
    
    def _build_middle_blocks(self):
        """Build middle blocks"""
        # Get the final channel count from downsampling
        final_ch = self.model_channels * self.channel_mult[-1]
        
        self.middle_block1 = ResidualBlock(final_ch, final_ch, self.time_embed_dim, 0.0)
        self.middle_attention = AttentionBlock(final_ch)
        self.middle_block2 = ResidualBlock(final_ch, final_ch, self.time_embed_dim, 0.0)
    
    def _build_up_blocks(self):
        """Build upsampling blocks"""
        self.up_blocks = nn.ModuleList()
        
        # Start with final channels from middle block
        ch = self.model_channels * self.channel_mult[-1]
        
        # Reverse the channel multipliers for upsampling
        for level, mult in enumerate(reversed(self.channel_mult)):
            out_ch = self.model_channels * mult
            
            for i in range(self.num_res_blocks + 1):
                # Calculate current resolution for attention
                current_res = self.spatial_size[0] // (2 ** (len(self.channel_mult) - 1 - level))
                has_attention = current_res in self.attention_resolutions
                
                # Get skip connection channels
                if len(self.down_channels) > 0:
                    skip_ch = self.down_channels.pop()
                else:
                    skip_ch = ch
                
                self.up_blocks.append(UpBlock(
                    ch, skip_ch, out_ch, self.time_embed_dim, has_attention, 0.0
                ))
                ch = out_ch

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
            #print("shape of condition: ",condition.shape)
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
        
        # Upsampling - use skip connections in reverse order
        for up_block in self.up_blocks:
            if len(skips) > 0:
                skip = skips.pop()
            else:
                # If no skip connection available, use current x
                skip = x
            x = up_block(x, skip, time_emb)
        
        # Final layers
        x = self.final_res_block(x, time_emb)
        x = self.final_conv(x)
        
        return x


# Improved wrapper class
class UNetWrapper(nn.Module):
    def __init__(self, unet, spatial_h, spatial_w, encode_t_dim, condition_dim=None):
        super().__init__()
        self.unet = unet
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.encode_t_dim = encode_t_dim
        self.condition_dim = condition_dim or 0
        
    def forward(self, x, time_steps=None, condition=None):
        """
        Forward pass supporting both 1D flattened input and proper 2D input
        
        Args:
            x: Input tensor
               - If 1D: [B, spatial_dim + time_dim + condition_dim]
               - If 2D: [B, C, H, W]
            time_steps: [B] time steps (optional, will be extracted from x if None)
            condition: [B, condition_dim] condition (optional, will be extracted from x if None)
        """
        batch_size = x.shape[0]
        #print("shape of input: ",x.shape, self.condition_dim)
        if len(x.shape) == 2:  # Flattened input
            # Parse flattened input
            spatial_dim = self.spatial_h * self.spatial_w
            spatial_data = x[:, :spatial_dim]
            remaining_data = x[:, spatial_dim:]
            
            # Extract time and condition
            if time_steps is None:
                time_data = remaining_data[:, :self.encode_t_dim]
                # Create time steps from time data (you may need to adjust this)
                time_steps = torch.norm(time_data, dim=1)
            
            if condition is None and self.condition_dim > 0:
                condition = remaining_data[:, self.encode_t_dim:self.encode_t_dim + self.condition_dim]
            
            # Reshape spatial data to 2D
            spatial_2d = spatial_data.view(batch_size, 1, self.spatial_h, self.spatial_w)
            
        else:  # Already 2D input
            spatial_2d = x
            if time_steps is None:
                time_steps = torch.zeros(batch_size, device=x.device)
        
        # Forward through UNet
        #print(f"shape of spatial_2d: {spatial_2d.shape}, time_steps: {time_steps.shape}, condition: {condition.shape}")
        output = self.unet(spatial_2d, time_steps, condition)
        
        # Return in same format as input
        if len(x.shape) == 2:
            return output.view(batch_size, -1)
        else:
            return output


# Factory function for building UNet subnet
def build_unet_subnet(encode_t_dim, dim_embedding, seq_len=None, layer_cond=False, 
                     spatial_size=(16, 9), params=None):
    """Build UNet-based subnet for diffusion/flow matching"""
    
    if params is None:
        params = {}
    
    # Calculate condition dimension
    #cond_dim = encode_t_dim + dim_embedding
    cond_dim=64
    if layer_cond and seq_len is not None:
        cond_dim += seq_len
    
    print(f"Building UNet with condition_dim={cond_dim}, spatial_size={spatial_size}")
    
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
    
    return UNetWrapper(unet, spatial_size[0], spatial_size[1], encode_t_dim, cond_dim)


# Example usage
if __name__ == "__main__":
    # Test the fixed implementation
    print("Testing UNet implementation...")
    
    # Parameters
    encode_t_dim = 64
    dim_embedding = 64
    seq_len = 45
    layer_cond = False
    spatial_size = (16, 9)
    
    # Calculate expected condition dimension
    expected_cond_dim = encode_t_dim + dim_embedding
    if layer_cond:
        expected_cond_dim += seq_len
    
    print(f"Expected condition dimension: {expected_cond_dim}")
    
    model = build_unet_subnet(
        encode_t_dim=encode_t_dim,
        dim_embedding=dim_embedding,
        seq_len=seq_len,
        layer_cond=layer_cond,
        spatial_size=spatial_size
    )
    
    # Test with flattened input
    spatial_dim = spatial_size[0] * spatial_size[1]
    total_dim = spatial_dim + expected_cond_dim #144+128 =272
    print(f"Creating input tensor with shape: [4, {total_dim}]")
    
    x = torch.randn(4, total_dim)
    print("shape of x: ",x.shape)
    out = model(x)
    print(f"Flattened input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test with 2D input
#     x_2d = torch.randn(4, 1, spatial_size[0], spatial_size[1])
#     time_steps = torch.randn(4)
#     condition = torch.randn(4, expected_cond_dim)
#     out_2d = model(x_2d, time_steps, condition)
#     print(f"2D input shape: {x_2d.shape}")
#     print(f"2D output shape: {out_2d.shape}")
    
#     print("All tests passed!")