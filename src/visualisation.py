import os
import torch
import matplotlib.pyplot as plt

from dataset import GalaxyDataset
from model import Decoder
from diffusion import Diffusion
from helpers import set_seed, generate_train_valid_datasets

# Configuration
TIMESTEPS_DIFF = 1000
MAX_SPARSITY = 0.20
MAX_NOISE_STD = 0.005
OUTPUT_DIR = "../results/ep200_sp0.20_std5e-3/visual"
MODEL_PATH = "../results/ep200_sp0.20_std5e-3/training/decoder_best.pt"
NUM_SAMPLES = 5

def generate_and_plot(dataset_subset, subset_name, decoder, diffusion, device):
    """
    Takes a dataset subset, generates reconstructions for the first NUM_SAMPLES,
    and plots them side-by-side.
    """
    for i in range(NUM_SAMPLES):
        x0, norms, num_gals = dataset_subset[i]

        # Add batch dimension and move to device
        x0 = x0.unsqueeze(0).to(device)
        norms = norms.unsqueeze(0).to(device)
        num_gals = num_gals.unsqueeze(0).to(device)

        with torch.no_grad():
            # Create starting degraded observation xT at t = 1000
            t_max = torch.full((1,), diffusion.timesteps, device=device, dtype=torch.long)
            xT = diffusion.degrade(x0, norms, num_gals, t_max)

            # Run Reverse Sampling
            x0_hat = diffusion.sample(decoder, xT, norms, num_gals)

        # Convert back to numpy for plotting
        x0_np = x0.cpu().numpy()[0, 0]      # Ploting the first channel
        xT_np = xT.cpu().numpy()[0, 0]
        x0_hat_np = x0_hat.cpu().numpy()[0, 0]

        # Plotting
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        
        # Fixed color bounds to avoid matplotlib scaling artifacts
        vmin, vmax = 0.0, 1.0

        ax[0].imshow(x0_np, cmap="viridis", vmin=vmin, vmax=vmax)
        ax[0].set_title(r"$x_0$ (true)")
        ax[0].axis("off")

        ax[1].imshow(xT_np, cmap="viridis", vmin=vmin, vmax=vmax)
        ax[1].set_title(r"$x_T$ (noised at t=1000)")
        ax[1].axis("off")

        ax[2].imshow(x0_hat_np, cmap="viridis", vmin=vmin, vmax=vmax)
        ax[2].set_title(r"$\hat{x}_0$ (reconstruction)")
        ax[2].axis("off")

        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(OUTPUT_DIR, f"{subset_name}_sample_{i+1}.png")
        plt.savefig(filepath, dpi=200)
        plt.close()
        
        print(f"Saved {filepath}")

def main():
    # Enforce exact same split as training
    set_seed(42)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    dataset = GalaxyDataset(
        data_folder="/scratch/izar/sarro/BAHAMAS-data-nonsparse/"
    )
    train_dataset, valid_dataset = generate_train_valid_datasets(dataset, frac=0.8)

    # Load Model & Diffusion
    diffusion = Diffusion(
        timesteps=TIMESTEPS_DIFF, 
        max_sparsity=MAX_SPARSITY, 
        max_noise_std=MAX_NOISE_STD, 
        device=device
    )
    
    decoder = Decoder(timesteps=TIMESTEPS_DIFF).to(device)
    decoder.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    decoder.eval()

    print(f"Generating {NUM_SAMPLES} training plots...")
    generate_and_plot(train_dataset, "train", decoder, diffusion, device)

    print(f"\nGenerating {NUM_SAMPLES} validation plots...")
    generate_and_plot(valid_dataset, "valid", decoder, diffusion, device)
    
    print("\nAll plots successfully generated!")

if __name__ == "__main__":
    main()