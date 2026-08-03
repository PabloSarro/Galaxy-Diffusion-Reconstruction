import os
import time
import torch
from torch.utils.data import DataLoader

from dataset import GalaxyDataset
from model import Decoder, cold_diffusion_loss
from diffusion import Diffusion
from helpers import (
    set_seed, 
    generate_train_valid_datasets, 
    plot_losses, 
    evaluate_cold_diffusion, 
    plot_cold_diffusion_reconstruction
)


# Cold Diffusion parameters
TIMESTEPS_DIFF = 1000 # Set to 10 for short runs
MAX_SPARSITY = 0.20 # 0.25 first training
MAX_NOISE_STD = 0.005 # 0.01 first training

# Training parameters
EPOCHS = 200 # Set to 5 for short runs
BATCH_SIZE = 64
LR = 1e-4
DEBUG = False # Set to True for short runs

# Output parameters
job_id = os.environ.get("SLURM_JOB_ID", "local") # Get the SLURM Job ID (local if sbatch not used)
OUTPUT_DIR = f"results_{job_id}"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "decoder_best.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True) # Create the directory for this run
print(f"All outputs for this run will be saved to: {OUTPUT_DIR}/")


set_seed(42)

# Dataset
dataset = GalaxyDataset(data_folder="/scratch/izar/sarro/BAHAMAS-data-nonsparse/")

train_dataset, valid_dataset = generate_train_valid_datasets(dataset, frac=0.8)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator().manual_seed(42), # Done to provide the optimizer with the same batch order across different runs, for better comparison.
    pin_memory=True
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, # No generator needed here, since there is no shuffle, and hence indices will be: [0, 1, 2, ...]
    pin_memory=True
)

# Model and Diffusion Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

diffusion = Diffusion(
    timesteps=TIMESTEPS_DIFF,
    max_sparsity=MAX_SPARSITY,
    max_noise_std=MAX_NOISE_STD,
    device=device
)

decoder = Decoder(timesteps=TIMESTEPS_DIFF).to(device)

optimizer = torch.optim.AdamW(
    decoder.parameters(),
    lr=LR,
    weight_decay=1e-4
)

# Training Loop
train_losses = []
valid_losses = []
best_loss = float("inf")

start = time.time()

for epoch in range(EPOCHS):
    total_train_loss = 0
    debug_print = True

    for x0, norms, num_gals in train_loader:
        x0 = x0.to(device, non_blocking=True)
        norms = norms.to(device, non_blocking=True)
        num_gals = num_gals.to(device, non_blocking=True)
        
        optimizer.zero_grad()

        # Sample diffusion timestep
        t = diffusion.sample_timesteps(x0.size(0))

        # Apply forward degradation
        x_t = diffusion.degrade(x0, norms, num_gals, t)

        # Predict image from degraded + timestep
        x0_pred = decoder(x_t, t)

        # Loss function between predicted map and true image
        loss = cold_diffusion_loss(x0_pred, x0)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
        optimizer.step()

        if DEBUG and debug_print:
            print(f"[DEBUG Epoch {epoch+1}]:")
            print(f"x_0 min          = {x0.min().item():.4f}, max={x0.max().item():.4f}, mean={x0.mean().item():.4f}")
            print(f"x_t min          = {x_t.min().item():.4f}, max={x_t.max().item():.4f}, mean={x_t.mean().item():.4f}")
            print(f"predicted_x0 min = {x0_pred.min().item():.4f}, max={x0_pred.max().item():.4f}")
            print(f"loss             = {loss.item():.4f}")
            debug_print = False # One DEBUG print per epoch.

        total_train_loss += loss.item()

    epoch_train_loss = total_train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    
    # Validation Loop
    decoder.eval()
    total_valid_loss = 0

    with torch.no_grad():
        for x0, norms, num_gals in valid_loader:
            x0 = x0.to(device, non_blocking=True)
            norms = norms.to(device, non_blocking=True)
            num_gals = num_gals.to(device, non_blocking=True)

            t = diffusion.sample_timesteps(x0.size(0))
            x_t = diffusion.degrade(x0, norms, num_gals, t)
            x0_pred = decoder(x_t, t)

            loss = cold_diffusion_loss(x0_pred, x0)
            total_valid_loss += loss.item()

    epoch_valid_loss = total_valid_loss / len(valid_loader)
    valid_losses.append(epoch_valid_loss)

    decoder.train()

    print(f"Epoch {epoch+1}/{EPOCHS} --> Train: {epoch_train_loss:.4f} | Valid: {epoch_valid_loss:.4f} (in {time.time()-start:.1f}s).")

    if epoch_valid_loss < best_loss:
        best_loss = epoch_valid_loss
        torch.save(decoder.state_dict(), BEST_MODEL_PATH)
        print("   Best model was stored")


# POST-TRAINING ANALYSIS

# Visualise plot of the training and validation losses after each epoch.
plot_losses(
    train_losses=train_losses,
    valid_losses=valid_losses,
    output_dir=OUTPUT_DIR
)
# Algo
evaluate_cold_diffusion(
    decoder=decoder, 
    diffusion=diffusion,
    device=device,
    valid_loader=valid_loader,
    max_batches=10,
    model_path=BEST_MODEL_PATH
)
# Algo
plot_cold_diffusion_reconstruction(
    decoder=decoder, 
    diffusion=diffusion,
    device=device,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    n_samples=5,
    model_path=BEST_MODEL_PATH,
    output_dir=OUTPUT_DIR
)