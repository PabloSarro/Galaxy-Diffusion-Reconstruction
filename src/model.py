import torch
import torch.nn as nn
import torch.nn.functional as F

# ENCODER
class Encoder(nn.Module):

    def __init__(self, latent_dim=128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # (B, 2, 100, 100) --> (B, 32, 100, 100)   # [(3*3)*2+1]*32 parameters
            nn.ReLU(),                                   # (B, 32, 100, 100) --> (B, 32, 100, 100)  # 0 parameters

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B, 32, 100, 100) --> (B, 64, 100, 100)  # [(3*3)*32+1]*64 parameters
            nn.ReLU(),                                   # (B, 64, 100, 100) --> (B, 64, 100, 100)  # 0 parameters

            nn.Flatten(),                                # (B, 64, 100, 100) --> (B, 64*100*100)    # 0 parameters

            nn.Linear(64 * 100 * 100, latent_dim)        # (B, 64*100*100) --> (B, latent_dim)      # (64*100*100+1)*latent_dim parameters
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

        # Project latent vector to feature maps
        self.z_proj = nn.Linear(
            latent_dim,
            256
        )

        # Timestep embedding layer
        # self.time_embedding = SinusoidalEmbedding(128)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,256)
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
            nn.Conv2d(192, 128, kernel_size=3, padding=1), # 128+64
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 2, kernel_size=1)
        )


    def forward(self, x_t, z, t):
        # Normalise embedding
        z = F.normalize(z, dim=1)                                   # (B, 128)

        # Project latent embedding
        z = self.z_proj(z)                                          # (B, 128) --> (B, 256)

        # Timestep embedding
        t = t.float().unsqueeze(1)                                  # (B, ) --> (B, 1)  (i.e. [3, 1, 2] --> [[3], [1], [2]] ).
        t = t / (self.timesteps-1)                                  # Normalise: [0, 1000] --> [0, 1]
        t = self.time_mlp(t)                                        # (B, 1) --> (B, 256)

        # Combine both conditionings
        cond = torch.cat([z, t], dim=1)                             # (B, 256) & (B, 256) --> (B, 512)
        cond = self.cond_proj(cond)                                 # (B, 512) --> (B, 256)
        
        # Condition by concatenation
        e1 = self.enc1(x_t)                                         # (B, 2, 100, 100) --> (B, 64, 100, 100)
        e2 = self.enc2(self.pool(e1))                               # (B, 64, 100, 100) --> (B, 64, 50, 50) --> (B, 128, 50, 50)
        b = self.bottleneck(e2)                                     # (B, 128, 50, 50) --> (B, 256, 50, 50)

        cond = cond[:, :, None, None]                               # (B, 256) --> (B, 256, 1, 1)
        cond = cond.expand(-1, -1, b.shape[2], b.shape[3])          # (B, 256, 1, 1) --> (B, 256, 50, 50)

        # Condition by concatenation
        b = torch.cat([b, cond], dim=1)                             # (B, 256, 50, 50) & (B, 256, 50, 50) --> (B, 512, 50, 50)

        # Inject latent at bottleneck
        d = self.up(b)                                              # (B, 512, 50, 50) --> (B, 128, 100, 100)
        d = torch.cat([d, e1], dim=1)                               # (B, 128, 100, 100) & (B, 64, 100, 100) --> (B, 192, 100, 100)

        noise_pred = self.dec(d)                                    # (B, 192, 100, 100) --> (B, 2, 100, 100)
        return noise_pred


def diffusion_loss(predicted_noise, true_noise):
    return F.mse_loss(predicted_noise, true_noise)