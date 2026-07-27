import time
import torch
from helpers import generate_train_valid_datasets
from torch.utils.data import DataLoader

from dataset import GalaxyDataset
from model import Encoder, Decoder, diffusion_loss
from diffusion import Diffusion
from helpers import set_seed, plot_losses, evaluate_reconstruction, plot_reconstruction_img

import torch.nn.functional as F


# Dataset noising parameters
NOISE_STD = 0.01
SPARSITY = 0.25
# Encoder parameters
LATENT_DIM = 128
ENCODER_PATH = "../results/dim128_0.01_0.25/encoder_best.pt"
# Diffusion parameters
TIMESTEPS_DIFF = 1000
# Training parameters
EPOCHS = 150
DEBUG = True

set_seed(42)

# Dataset
dataset = GalaxyDataset(
    data_folder="/scratch/izar/sarro/BAHAMAS-data-nonsparse/",
    noise_std=NOISE_STD,
    sparsity=SPARSITY
)

train_dataset, valid_dataset = generate_train_valid_datasets(dataset, frac=0.8)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    generator=torch.Generator().manual_seed(42), # Done to provide the optimizer with the same batch order across different runs, for better comparison.
    pin_memory=True
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=64,
    shuffle=False, # No generator needed here, since there is no shuffle, and hence indices will be: [0, 1, 2, ...]
    pin_memory=True
)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoder
encoder = Encoder(latent_dim=LATENT_DIM).to(device)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
encoder.eval() # Use encoder for evaluation (training already done in train_encoder.py)

for p in encoder.parameters():
    p.requires_grad = False

# Decoder
diffusion = Diffusion(timesteps=TIMESTEPS_DIFF, device=device)
decoder = Decoder(timesteps=TIMESTEPS_DIFF, latent_dim=LATENT_DIM).to(device)

optimizer = torch.optim.AdamW(
    decoder.parameters(),
    lr=1e-5,
    weight_decay=1e-4
)

# Training
epochs = EPOCHS
train_losses = []
valid_losses = []
best_loss = float("inf")

start = time.time()

for epoch in range(epochs):
    total_train_loss = 0
    debug_print = True

    for TS, NS in train_loader:
        TS = TS.to(device, non_blocking=True)
        NS = NS.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        with torch.no_grad():
            z_obs = encoder(NS) # torch.zeros(TS.size(0), 128, device=device)

        t = diffusion.sample_timesteps(TS.size(0))   # Sample diffusion timestep
        x_t, noise = diffusion.q_sample(TS, t)       # Add diffusion noise
        noise_pred = decoder(x_t, z_obs, t)          # Predict noise

        loss = diffusion_loss(noise_pred, noise)     # Loss function
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
        optimizer.step()

        if DEBUG and debug_print:
            print(f"[DEBUG]:")
            print(f"TS min         = {TS.min().item()},         max={TS.max().item()},         mean={TS.mean().item()},         std={TS.std().item()}")
            print(f"z_obs min      = {z_obs.min().item()},      max={z_obs.max().item()},      mean={z_obs.mean().item()},      std={z_obs.std().item()},     norm={z_obs.norm(dim=1).mean()}, abs mean={z_obs.abs().mean()}")
            print(f"noise min      = {noise.min().item()},      max={noise.max().item()},      mean={noise.mean().item()},      std={noise.std().item()}")
            print(f"noise_pred min = {noise_pred.min().item()}, max={noise_pred.max().item()}, mean={noise_pred.mean().item()}, std={noise_pred.std().item()}")
            print(f"loss = {loss}")
            debug_print = False # One DEBUG print per epoch.

        total_train_loss += loss.item()

    epoch_train_loss = total_train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    
    # COMPUTE VALIDATION LOSS
    decoder.eval()
    total_valid_loss = 0

    with torch.no_grad():

        for TS, NS in valid_loader:
            TS = TS.to(device, non_blocking=True)
            NS = NS.to(device, non_blocking=True)

            z_obs = encoder(NS) # torch.zeros(TS.size(0), 128, device=device)

            t = diffusion.sample_timesteps(TS.size(0))
            x_t, noise = diffusion.q_sample(TS, t)
            noise_pred = decoder(x_t, z_obs, t)

            loss = diffusion_loss(noise_pred, noise)

            total_valid_loss += loss.item()

    epoch_valid_loss = total_valid_loss / len(valid_loader)
    valid_losses.append(epoch_valid_loss)

    # Conditional Reconstruction Diagnostic
    if epoch % 10 == 0 or epoch == EPOCHS-1:
        with torch.no_grad():
            TS, NS = next(iter(valid_loader))
            TS = TS.to(device)
            NS = NS.to(device)
            z_obs = encoder(NS)
            # Random embedding, for diagnosing
            z_random = torch.randn_like(z_obs)
            recon_real = diffusion.sample(decoder, z_obs[:4], TS[:4].shape)
            recon_random = diffusion.sample(decoder, z_random[:4], TS[:4].shape)
            mse_real = F.mse_loss(recon_real, TS[:4])
            mse_random = F.mse_loss(recon_random, TS[:4])
            print(f"DIAGNOSTIC: MSE real={mse_real:.4f}, MSE random={mse_random:.4f}")

    decoder.train()

    print(
        f"Epoch {epoch+1}/{epochs} --> Train: {epoch_train_loss:.4f} | Valid: {epoch_valid_loss:.4f} (in {time.time()-start:.1f}s)."
    )

    if epoch_valid_loss < best_loss:
        best_loss = epoch_valid_loss
        torch.save(
            decoder.state_dict(),
            "decoder_best.pt"
        )
        print("   Best model was stored")



# Visualise plot of the training and validation losses after each epoch.
plot_losses(epochs, train_losses, valid_losses)

evaluate_reconstruction(encoder, decoder, diffusion, device, valid_loader, max_batches=10)
# Low MSE + high Pearson → good reconstruction.
# Low MSE + low Pearson → blurry/mean reconstruction.
# High Pearson + high MSE → correct structures but wrong amplitudes.

plot_reconstruction_img(encoder, decoder, diffusion, device, valid_dataset)