import os
import torch

from dataset import GalaxyDataset
from model import Decoder
from diffusion import Diffusion
from helpers import (
    set_seed,
    generate_train_valid_datasets,
    generate_and_plot,
)

# ==========================================================
# Configuration
# ==========================================================

TIMESTEPS_DIFF = 1000
MAX_SPARSITY = 0.20
MAX_NOISE_STD = 0.005

OUTPUT_DIR = "../results/new_arch/ep50_sp0.2_std0.005"
TRAINING_DIR = os.path.join(OUTPUT_DIR, "training")
VISUAL_DIR = os.path.join(OUTPUT_DIR, "visual")

BEST_MODEL_PATH = os.path.join(TRAINING_DIR, "decoder_best.pt")

N_SAMPLES = 5

# ==========================================================
# Setup
# ==========================================================

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# Dataset
# ==========================================================

dataset = GalaxyDataset(data_folder="/scratch/izar/sarro/BAHAMAS-data-nonsparse/")
train_dataset, valid_dataset = generate_train_valid_datasets(dataset, frac=0.8)

# ==========================================================
# Diffusion
# ==========================================================

diffusion = Diffusion(
    timesteps=TIMESTEPS_DIFF,
    max_sparsity=MAX_SPARSITY,
    max_noise_std=MAX_NOISE_STD,
    device=device
)

# ==========================================================
# Model
# ==========================================================

decoder = Decoder(timesteps=TIMESTEPS_DIFF).to(device)
decoder.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
decoder.eval()

# ==========================================================
# Visualisation
# ==========================================================

os.makedirs(VISUAL_DIR, exist_ok=True)

print("Generating training examples...")
generate_and_plot(
    decoder=decoder,
    diffusion=diffusion,
    device=device,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    n_samples=N_SAMPLES,
    model_path=BEST_MODEL_PATH,
    output_dir=VISUAL_DIR
)