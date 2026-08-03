import os
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt


# Deterministic Runs for Better Comparison
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Dataset Splitting
def generate_train_valid_datasets(dataset, frac=0.8):
    start = time.time()

    all_indices = np.arange(len(dataset))
    cross_sections = dataset.cross_sections

    train_indices = []
    valid_indices = []

    for sigma in np.unique(cross_sections):

        # All simulations with this cross-section
        sigma_indices = all_indices[cross_sections == sigma]

        # Shuffle within this cross-section
        np.random.shuffle(sigma_indices)
        n_train_sigma = int(frac * len(sigma_indices))
        train_indices.extend(sigma_indices[:n_train_sigma])
        valid_indices.extend(sigma_indices[n_train_sigma:])

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(dataset, valid_indices)

    end = time.time()
    print(f"Train+Valid datasets generated (in {end-start:.5f}s).")
    
    return train_dataset, valid_dataset


# Loss Plotting
def plot_losses(train_losses, valid_losses, output_dir):
    epochs = len(train_losses)

    plt.figure(figsize=(6,4))

    plt.plot(range(1, epochs+1), train_losses, label="Train")
    plt.plot(range(1, epochs+1), valid_losses, label="Valid")

    plt.legend(loc="best")
    plt.xlabel("Epoch")
    plt.ylabel("Cold Diffusion Loss (MSE)")
    plt.title("Training and Validation Losses")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_valid_losses.png"), dpi=200)
    plt.close()


# Check MSE + Pearson
def evaluate_cold_diffusion(decoder, diffusion, device, valid_loader, max_batches=None, model_path="decoder_best.pt"):
    """
    Evaluate Cold Diffusion reconstruction quality (MSE & Pearson).
    """
    start = time.time()

    decoder.load_state_dict(torch.load(model_path, map_location=device))
    decoder.eval()

    mse_values = []
    corr_values = []
    batches = 0

    with torch.no_grad():
        for x0, norms, num_gals in valid_loader:
            x0 = x0.to(device)
            norms = norms.to(device)
            num_gals = num_gals.to(device)

            # Generate fully degraded observation at t=1000.
            t_max = torch.full((x0.size(0),), diffusion.timesteps, device=device, dtype=torch.long)
            xT = diffusion.degrade(x0, norms, num_gals, t_max)

            # Run Cold Diffusion reverse sampling starting from xT
            x0_hat = diffusion.sample(decoder, xT, norms, num_gals)

            # MSE
            mse = torch.mean((x0 - x0_hat)**2)
            mse_values.append(mse.item())

            # Pearson correlation
            x0_flat = x0.flatten(1)
            x0_hat_flat = x0_hat.flatten(1)

            x0_centered = x0_flat - x0_flat.mean(dim=1, keepdim=True)
            x0_hat_centered = x0_hat_flat - x0_hat_flat.mean(dim=1, keepdim=True)

            eps = 1e-8
            cov = (x0_centered * x0_hat_centered).sum(dim=1)
            std_prod = torch.sqrt((x0_centered**2).sum(dim=1))*torch.sqrt((x0_hat_centered**2).sum(dim=1))
            
            corr = cov / (std_prod + eps)
            corr_values.extend(corr.cpu().numpy())

            batches += 1
            if max_batches is not None and batches >= max_batches:
                break

    end = time.time()
    print(f"\n=== Cold Diffusion Evaluation Summary ===")
    print(f"MSE     : {np.mean(mse_values):.6f}")
    print(f"Pearson : {np.mean(corr_values):.4f}")
    print(f"MSE + Pearson Evaluation took {end-start:.1f}s.\n")



# Cold Diffusion Reconstruction Visualisation
def plot_cold_diffusion_reconstruction(decoder, diffusion, device, train_dataset, valid_dataset, n_samples=5, model_path="decoder_best.pt", output_dir="."):
    start = time.time()

    decoder.load_state_dict(torch.load(model_path, map_location=device))
    decoder.eval()

    datasets = [("train", train_dataset), ("valid", valid_dataset)]
    os.makedirs(output_dir, exist_ok=True)

    for prefix, dataset in datasets:
        for i in range(n_samples):
            x0, norms, num_gals = dataset[i]

            x0 = x0.unsqueeze(0).to(device)
            norms = norms.unsqueeze(0).to(device)
            num_gals = num_gals.unsqueeze(0).to(device)

            with torch.no_grad():
                # Create starting degraded observation NS at t = 1000
                t_max = torch.full((1,), diffusion.timesteps, device=device, dtype=torch.long)
                xT = diffusion.degrade(x0, norms, num_gals, t_max)

                # Run Reverse Sampling
                x0_hat = diffusion.sample(decoder, xT, norms, num_gals)

            x0_np = x0.cpu().numpy()[0]
            xT_np = xT.cpu().numpy()[0]
            x0_hat_np = x0_hat.cpu().numpy()[0]

            fig, ax = plt.subplots(1,3, figsize=(12,4))
            vmin, vmax = 0.0, 1.0

            ax[0].imshow(x0_np[0], cmap="viridis", vmin=vmin, vmax=vmax)
            ax[0].set_title(r"$x_0$ (true)")
            ax[0].axis("off")

            ax[1].imshow(xT_np[0], cmap="viridis", vmin=vmin, vmax=vmax)
            ax[1].set_title(rf"$x_T$ (noised at t={diffusion.timesteps})")
            ax[1].axis("off")

            ax[2].imshow(x0_hat_np[0], cmap="viridis", vmin=vmin, vmax=vmax)
            ax[2].set_title(r"$\hat{x}_0$ (reconstruction)")
            ax[2].axis("off")

            plt.tight_layout()

            filepath = os.path.join(output_dir, f"{prefix}_sample_{i+1}.png")
            plt.savefig(filepath, dpi=200)
            plt.close()

    end = time.time()
    total_plots = len(datasets) * n_samples
    print(f"Plotting all {total_plots} reconstructed images took {end-start:.1f}s.")