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


# DECODER
class Decoder(nn.Module):

    def __init__(self, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps

        # Timestep embedding layer
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        # U-Net encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.up = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2)

        self.dec = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1), # 128 (up) + 64 (enc1)
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1)
        )


    def forward(self, x_t, t):
        # Timestep embedding
        t_emb = self.time_mlp(t)                                    # (B,) --> (B, 256)

        # Encoder path
        e1 = self.enc1(x_t)                                         # (B, 2, 100, 100) --> (B, 64, 100, 100)
        e2 = self.enc2(self.pool(e1))                               # (B, 64, 100, 100) --> (B, 64, 50, 50) --> (B, 128, 50, 50)

        b = self.bottleneck(e2)                                     # (B, 128, 50, 50) --> (B, 256, 50, 50)

        # Condition by concatenation at the bottleneck
        t_emb = t_emb[:, :, None, None]                             # (B, 256) --> (B, 256, 1, 1)
        t_emb = t_emb.expand(-1, -1, b.shape[2], b.shape[3])        # (B, 256, 1, 1) --> (B, 256, 50, 50)

        b = torch.cat([b, t_emb], dim=1)                            # (B, 256, 50, 50) & (B, 256, 50, 50) --> (B, 512, 50, 50)

        # Decoder path
        d = self.up(b)                                              # (B, 512, 50, 50) --> (B, 128, 100, 100)
        d = torch.cat([d, e1], dim=1)                               # (B, 128, 100, 100) & (B, 64, 100, 100) --> (B, 192, 100, 100)

        # Predict the new image
        x0_pred = self.dec(d)                                       # (B, 192, 100, 100) --> (B, 2, 100, 100)
        return x0_pred


def cold_diffusion_loss(predicted_x0, true_x0):
    """
    Cold Diffusion: non-Gaussian noise schedule, where the network
    predicts the clean image directly, instead of the added noise.
    """
    return F.mse_loss(predicted_x0, true_x0)