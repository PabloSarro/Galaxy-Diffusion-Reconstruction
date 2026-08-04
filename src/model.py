import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# SINUSOIDAL POSITION EMBEDDING (for t)
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Standard Fourier features for timestep embedding.
    Maps integer timestep t to a higher-dimensional space.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class SelfAttention(nn.Module):
    """
    Standard Self-Attention block to help the model learn global spatial dependencies.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        
        h = self.norm(x).view(B, C, H*W).transpose(1,2)         # GroupNorm, reshape as: (B, C, H, W) -> (B, H*W, C).
        att_output, _ = self.mha(h, h, h)                       # Self-attention.
        att_output = att_output.transpose(1,2).view(B, C, H, W) # Reshape back and add residual skip connection.

        return x + att_output


class ResnetBlock(nn.Module):
    """
    A modern Diffusion ResNet block:
    GroupNorm -> SiLU -> Conv -> Add Time Embedding -> GroupNorm -> SiLU -> Conv
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.shortcut = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        # First block
        h = self.block1(x)
        # Inject Time Embedding
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb
        # Second block
        h = self.block2(h)
        # Add shortcut skip connection
        return h + self.shortcut(x)


# DECODER
class Decoder(nn.Module):
    def __init__(self, timesteps=1000, in_channels=2, out_channels=2, base_channels=64):
        super().__init__()
        self.timesteps = timesteps

        # Time Embedding Dimensions
        time_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # =============== U-NET ENCODER ===============
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        # (Downsampling)
        self.down1 = ResnetBlock(base_channels, base_channels, time_dim)
        self.down2 = ResnetBlock(base_channels, base_channels*2, time_dim)
        self.down3 = ResnetBlock(2*base_channels, 4*base_channels, time_dim)

        # ============ ATTENTION (LOWER LAYERS) ============
        self.att_down = SelfAttention(4*base_channels)
        self.pool = nn.MaxPool2d(2)

        # =============== BOTTLENECK ===============
        self.mid1 = ResnetBlock(4*base_channels, 4*base_channels, time_dim)
        self.mid_att = SelfAttention(4*base_channels)
        self.mid2 = ResnetBlock(4*base_channels, 4*base_channels, time_dim)

        # ================= DECODER =================
        # (Upsampling)
        self.up1 = nn.ConvTranspose2d(4*base_channels, 2*base_channels, kernel_size=2, stride=2)
        self.att_up = SelfAttention(4*base_channels)
        self.up_res1 = ResnetBlock(4*base_channels, 2*base_channels, time_dim) # *4 due to skip connection concatenation

        self.up2 = nn.ConvTranspose2d(2*base_channels, base_channels, kernel_size=2, stride=2)
        self.up_res2 = ResnetBlock(2*base_channels, base_channels, time_dim)

        # Final projection to match the original shape dimensions (B, C, H, W)
        self.final_res = ResnetBlock(2*base_channels, base_channels, time_dim)
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, t):
        # Timestep embedding
        t_emb = self.time_mlp(t)                       # (B,) --> (B, 256)

        # Initial Convolution
        x = self.init_conv(x)                          # (B, 2, 100, 100) --> (B, 64, 100, 100)

        # Downsampling Path
        d1 = self.down1(x, t_emb)           # Skip 1
        x_pool1 = self.pool(d1)

        d2 = self.down2(x_pool1, t_emb)     # Skip 2
        x_pool2 = self.pool(d2)

        d3 = self.down3(x_pool2, t_emb)     # Skip 3
        d3 = self.att_down(d3)

        # Bottleneck
        b = self.mid1(d3, t_emb)
        b = self.mid_att(b)
        b = self.mid2(b, t_emb)

        # Upsampling Path
        up1 = self.up1(b)
        up1 = torch.cat([up1, d2], dim=1)   # Concatenate Skip 2
        up1 = self.att_up(up1)
        up1 = self.up_res1(up1, t_emb)

        up2 = self.up2(up1)
        up2 = torch.cat([up2, d1], dim=1)   # Concatenate Skip 1
        up2 = self.up_res2(up2, t_emb)

        # Final Convolution
        out = torch.cat([up2, x], dim=1)
        out = self.final_res(out, t_emb)
        return self.final_conv(out)


def cold_diffusion_loss(predicted_x0, true_x0):
    """
    Cold Diffusion: non-Gaussian noise schedule, where the network
    predicts the clean image directly, instead of the added noise.
    """
    return F.mse_loss(predicted_x0, true_x0)