import os
import time
import torch
from torch.utils.data import DataLoader

from dataset import GalaxyDataset
from model import Decoder, decoder_loss
from degradation import Degradation
from helpers import (
    set_seed, 
    generate_train_valid_datasets, 
    plot_losses, 
    evaluate_cold_diffusion, 
    generate_and_plot
)

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--loss", type=str, default="MSE", choices=["MSE", "MSE_MAE", "MSE_Grad", "Charbonnier", "Weighted"])
parser.add_argument("--alpha",type=float, default=1.0)

args = parser.parse_args()


# Degradation parameters
SPARSITY = 0.70
NOISE_STD = 0.005 # Still to check what the 0.26 represents!

# Training parameters
EPOCHS = 50 # Set to 2/3 for short runs
BATCH_SIZE = 16
LR = 1e-4
DEBUG = False # Set to True for short runs
LOSS_METHOD = args.loss
ALPHA = args.alpha

# Output parameters
PREFIX = "../results/5-low_resolution"
OUTPUT_DIR = os.path.join(PREFIX, LOSS_METHOD)

TRAINING_DIR = os.path.join(OUTPUT_DIR, "training")
VISUAL_DIR = os.path.join(OUTPUT_DIR, "visual")
BEST_MODEL_PATH = os.path.join(TRAINING_DIR, "decoder_best.pt")

os.makedirs(TRAINING_DIR, exist_ok=True) # Create the directory for this run
os.makedirs(VISUAL_DIR, exist_ok=True) # Create the directory for this run

print("=====================================================")
print("============= CONFIGURATION FOR THE RUN =============")
print("=====================================================\n")
print(f"Sparsity: {SPARSITY} | Noise std: {NOISE_STD}")
print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | lr: {LR}")
print(f"Loss: {LOSS_METHOD} | Alpha: {ALPHA}")
print(f"Storage path: {OUTPUT_DIR}\n")


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

degradation = Degradation(
    sparsity=SPARSITY,
    noise_std=NOISE_STD,
    device=device
)

decoder = Decoder().to(device)

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

        # Degrade the true image
        x_noised = degradation.degrade(x0, norms, num_gals)

        # Predict image from degraded image.
        x0_pred = decoder(x_noised)

        # Loss function between predicted map and true image
        loss = decoder_loss(x0_pred, x0, method=LOSS_METHOD, alpha=ALPHA)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
        optimizer.step()

        if DEBUG and debug_print:
            print(f"[DEBUG Epoch {epoch+1}]:")
            print(f"x_0 min          = {x0.min().item():.4f}, max={x0.max().item():.4f}, mean={x0.mean().item():.4f}")
            print(f"x_noised min     = {x_noised.min().item():.4f}, max={x_noised.max().item():.4f}, mean={x_noised.mean().item():.4f}")
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

            x_t = degradation.degrade(x0, norms, num_gals)
            x0_pred = decoder(x_t)

            loss = decoder_loss(x0_pred, x0, method=LOSS_METHOD, alpha=ALPHA)
            total_valid_loss += loss.item()

    epoch_valid_loss = total_valid_loss / len(valid_loader)
    valid_losses.append(epoch_valid_loss)

    decoder.train()

    print(f"Epoch {epoch+1}/{EPOCHS} --> Train: {epoch_train_loss:.6f} | Valid: {epoch_valid_loss:.6f} (in {time.time()-start:.1f}s).")

    if epoch_valid_loss < best_loss:
        best_loss = epoch_valid_loss
        torch.save(decoder.state_dict(), BEST_MODEL_PATH)
        print("   Best model was stored")


# POST-TRAINING ANALYSIS

# Visualise plot of the training and validation losses after each epoch.
plot_losses(
    train_losses=train_losses,
    valid_losses=valid_losses,
    loss_function=LOSS_METHOD,
    output_dir=TRAINING_DIR
)
# Return the MSE and Pearson values for 20 reconstructed images.
evaluate_cold_diffusion(
    decoder=decoder, 
    degradation=degradation,
    device=device,
    valid_loader=valid_loader,
    max_batches=20,
    model_path=BEST_MODEL_PATH
)
# Plot reconstruction for 5 images in the training & validation datasets.
generate_and_plot(
    decoder=decoder,
    degradation=degradation,
    device=device,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    n_samples=5,
    model_path=BEST_MODEL_PATH,
    output_dir=VISUAL_DIR
)