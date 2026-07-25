import torch
import torch.nn as nn
import torch.nn.functional as F

# ENCODER
class Encoder(nn.Module):

    def __init__(self, latent_dim=128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(64 * 100 * 100, latent_dim)
        )

    def forward(self, x):
        z = self.cnn(x)
        return z


def contrastive_loss(z_true, z_obs, temperature=0.07):
    """
    InfoNCE loss (taken from the CLIP paper: https://arxiv.org/pdf/2103.00020).

    z_true : (B, D)
    z_obs  : (B, D)
    """

    # L2-normalize embeddings
    z_true = F.normalize(z_true, dim=1)
    z_obs  = F.normalize(z_obs, dim=1)

    # Pairwise cosine similarities
    logits = torch.matmul(z_true, z_obs.T) / temperature

    # Correct pairs lie on the diagonal
    labels = torch.arange(
        z_true.size(0),
        device=z_true.device
    )

    # Symmetric CLIP loss
    loss_true = F.cross_entropy(logits, labels)
    loss_obs  = F.cross_entropy(logits.T, labels)

    return (loss_true + loss_obs) / 2


# DECODER
class Decoder(nn.Module):

    def __init__(self, timesteps=1000, latent_dim=128):
        super().__init__()

        self.timesteps = timesteps

        # Timestep embedding layer
        # self.time_embedding = SinusoidalEmbedding(128)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

        # Project latent vector to feature maps
        self.z_proj = nn.Linear(
            latent_dim,
            256
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,256)
        )

        # U-Net encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.up = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1), # 128+64
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 2, 1)
        )


    def forward(self, x_t, z, t):
        # Project latent embedding
        z = self.z_proj(z)              # (B, 256)

        # Timestep embedding
        t = t.float().unsqueeze(1)
        t = t / (self.timesteps-1)
        t = self.time_mlp(t)

        # Combine both conditionings
        cond = torch.cat([z, t], dim=1)
        cond = self.cond_proj(cond)
        
        # Condition by concatenation
        e1 = self.enc1(x_t)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(e2)

        cond = cond[:, :, None, None]
        cond = cond.expand(-1, -1, b.shape[2], b.shape[3])

        b = b + cond

        # Inject latent at bottleneck
        d = self.up(b)
        d = torch.cat([d, e1], dim=1)

        noise_pred = self.dec(d)
        return noise_pred


def diffusion_loss(predicted_noise, true_noise):
    return F.mse_loss(predicted_noise, true_noise)


# # SINUSOIDAL EMBEDDING
# class SinusoidalEmbedding(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, t):
#         half_dim = self.dim // 2

#         emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
#         emb = torch.exp(
#             torch.arange(
#                 half_dim,
#                 device=t.device
#             ) * -emb
#         )

#         emb = t[:, None] * emb[None, :]
#         emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

#         return emb