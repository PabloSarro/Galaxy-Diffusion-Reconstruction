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
from training import run_training


parser = argparse.ArgumentParser()
parser.add_argument("--loss", type=str, default="MSE", choices=["MSE", "MSE_MAE", "MSE_Grad", "Charbonnier", "Weighted"])
parser.add_argument("--alpha",type=float, default=1.0)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--output-prefix", default="../results/7-with_shape_noise",
                    help="Run folder root. Default is the reported Model 3 folder: do not point test runs there.")
parser.add_argument("--resume", action="store_true",
                    help="Continue from <run>/training/last.pt if it exists (else start fresh).")

args = parser.parse_args()


# Degradation parameters
SPARSITY = 0.70
NOISE_STD = 0.22 # 0.005 too optimistic # 0 to try

# Training parameters
EPOCHS = args.epochs # default 200; use a smaller value for short runs
BATCH_SIZE = 32
LR = 5e-4
DEBUG = False # Set to True for short runs
LOSS_METHOD = args.loss
ALPHA = args.alpha

# Output parameters
PREFIX = args.output_prefix
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
dataset = GalaxyDataset(data_folder="../data-full/")

train_dataset, valid_dataset = generate_train_valid_datasets(dataset, frac=0.8)
LOADER_GEN = torch.Generator().manual_seed(42)  # saved in last.pt so resumed runs see the same batch order

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=LOADER_GEN, # Done to provide the optimizer with the same batch order across different runs, for better comparison.
    pin_memory=True,
    num_workers=1
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, # No generator needed here, since there is no shuffle, and hence indices will be: [0, 1, 2, ...]
    pin_memory=True,
    num_workers=1
)

# Model and Diffusion Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"USING DEVICE: {device}")

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

# Training Loop (checkpointed; see src/training.py)
train_losses, valid_losses, best_loss = run_training(
    decoder=decoder,
    optimizer=optimizer,
    degradation=degradation,
    train_loader=train_loader,
    valid_loader=valid_loader,
    loader_gen=LOADER_GEN,
    epochs=EPOCHS,
    out_dir=TRAINING_DIR,
    loss_method=LOSS_METHOD,
    alpha=ALPHA,
    device=device,
    resume=args.resume,
)


# POST-TRAINING ANALYSIS

# Load the best model weights
decoder.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
decoder.eval()

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