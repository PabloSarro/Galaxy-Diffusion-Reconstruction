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
    plt.savefig(os.path.join(output_dir, "losses_train_valid.png"), dpi=200)
    plt.close()


# Check MSE + Pearson
def evaluate_cold_diffusion(decoder, diffusion, device, valid_loader, max_batches=None, model_path="decoder_best.pt"):
    """
    Evaluate Cold Diffusion reconstruction quality (MSE & Pearson).
    """
    start = time.time()

    decoder.load_state_dict(torch.load(model_path, map_location=device))
    decoder.eval()

    identity_mse_values = []
    direct_mse_values = []
    algo1_mse_values = []
    algo2_mse_values = []

    identity_corr_values = []
    direct_corr_values = []
    algo1_corr_values = []
    algo2_corr_values = []
    
    batches = 0

    with torch.no_grad():
        for x0, norms, num_gals in valid_loader:
            x0 = x0.to(device)
            norms = norms.to(device)
            num_gals = num_gals.to(device)

            # Generate fully degraded observation at t=1000.
            t_max = torch.full((x0.size(0),), diffusion.timesteps, device=device, dtype=torch.long)
            xT = diffusion.degrade(x0, norms, num_gals, t_max)

            # Ensure fair comparison between methods.
            B,C,H,W = xT.shape

            fixed_mask_rand = torch.rand((B,H,W), device=device)
            fixed_e1 = torch.randn((B,H,W), device=device)
            fixed_e2 = torch.randn((B,H,W), device=device)

            # Reconstructed images
            x0_direct = diffusion.sample(decoder, xT, norms, num_gals, method="direct", fixed_mask_rand=fixed_mask_rand, fixed_e1=fixed_e1, fixed_e2=fixed_e2)
            x0_algo1 = diffusion.sample(decoder, xT, norms, num_gals, method="algo1", fixed_mask_rand=fixed_mask_rand, fixed_e1=fixed_e1, fixed_e2=fixed_e2)
            x0_algo2 = diffusion.sample(decoder, xT, norms, num_gals, method="algo2", fixed_mask_rand=fixed_mask_rand, fixed_e1=fixed_e1, fixed_e2=fixed_e2)

            # MSEs
            identity_mse = torch.mean((x0 - xT)**2)
            direct_mse = torch.mean((x0 - x0_direct)**2)
            algo1_mse = torch.mean((x0 - x0_algo1)**2)
            algo2_mse = torch.mean((x0 - x0_algo2)**2)
            
            identity_mse_values.append(identity_mse.item())
            direct_mse_values.append(direct_mse.item())
            algo1_mse_values.append(algo1_mse.item())
            algo2_mse_values.append(algo2_mse.item())

            # Pearsons
            x0_flat = x0.flatten(1)
            xT_flat = xT.flatten(1)
            x0_direct_flat = x0_direct.flatten(1)
            x0_algo1_flat = x0_algo1.flatten(1)
            x0_algo2_flat = x0_algo2.flatten(1)

            x0_centered = x0_flat - x0_flat.mean(dim=1, keepdim=True)
            xT_centered = xT_flat - xT_flat.mean(dim=1, keepdim=True)
            x0_direct_centered = x0_direct_flat - x0_direct_flat.mean(dim=1, keepdim=True)
            x0_algo1_centered = x0_algo1_flat - x0_algo1_flat.mean(dim=1, keepdim=True)
            x0_algo2_centered = x0_algo2_flat - x0_algo2_flat.mean(dim=1, keepdim=True)

            eps = 1e-8

            identity_cov = (x0_centered * xT_centered).sum(dim=1)
            direct_cov = (x0_centered * x0_direct_centered).sum(dim=1)
            algo1_cov = (x0_centered * x0_algo1_centered).sum(dim=1)
            algo2_cov = (x0_centered * x0_algo2_centered).sum(dim=1)
            
            identity_std_prod = torch.sqrt((x0_centered**2).sum(dim=1))*torch.sqrt((xT_centered**2).sum(dim=1))
            direct_std_prod = torch.sqrt((x0_centered**2).sum(dim=1))*torch.sqrt((x0_direct_centered**2).sum(dim=1))
            algo1_std_prod = torch.sqrt((x0_centered**2).sum(dim=1))*torch.sqrt((x0_algo1_centered**2).sum(dim=1))
            algo2_std_prod = torch.sqrt((x0_centered**2).sum(dim=1))*torch.sqrt((x0_algo2_centered**2).sum(dim=1))
            
            identity_corr = identity_cov / (identity_std_prod + eps)
            direct_corr = direct_cov / (direct_std_prod + eps)
            algo1_corr = algo1_cov / (algo1_std_prod + eps)
            algo2_corr = algo2_cov / (algo2_std_prod + eps)

            identity_corr_values.extend(identity_corr.cpu().numpy())
            direct_corr_values.extend(direct_corr.cpu().numpy())
            algo1_corr_values.extend(algo1_corr.cpu().numpy())
            algo2_corr_values.extend(algo2_corr.cpu().numpy())

            batches += 1
            if max_batches is not None and batches >= max_batches:
                break

    end = time.time()
    print(f"\n=== Cold Diffusion Evaluation Summary ===")

    print("\n1. Baseline (Worst-Case): xT vs. x0")
    print(f"MSE     : {np.mean(identity_mse_values):.6f}")
    print(f"Pearson : {np.mean(identity_corr_values):.4f}")

    print("\n2. Direct Prediction: xT -> \hat[x0]")
    print(f"MSE     : {np.mean(direct_mse_values):.6f}")
    print(f"Pearson : {np.mean(direct_corr_values):.4f}")

    print("\n3. Diffusion Sampling (Algorithm 1)")
    print(f"MSE     : {np.mean(algo1_mse_values):.6f}")
    print(f"Pearson : {np.mean(algo1_corr_values):.4f}")

    print("\n4. Diffusion Sampling (Algorithm 2)")
    print(f"MSE     : {np.mean(algo2_mse_values):.6f}")
    print(f"Pearson : {np.mean(algo2_corr_values):.4f}")

    print(f"MSE + Pearson Evaluation took {end-start:.1f}s.\n")



def generate_and_plot(decoder, diffusion, device, train_dataset, valid_dataset, n_samples=5, model_path=None, output_dir=None):
    """
    Plot:
        True x0 | xT | Direct | Algorithm 1 | Algorithm 2
    """
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

                # Fully degraded image
                t_max = torch.full((1,), diffusion.timesteps, device=device, dtype=torch.long)
                xT = diffusion.degrade(x0, norms, num_gals, t_max)

                # Same randomness for both diffusion algorithms
                B, C, H, W = xT.shape
                fixed_mask_rand = torch.rand((B, H, W), device=device)
                fixed_e1 = torch.randn((B, H, W), device=device)
                fixed_e2 = torch.randn((B, H, W), device=device)

                # Reconstructions
                x_direct = diffusion.sample(decoder, xT, norms, num_gals, method="direct", fixed_mask_rand=fixed_mask_rand, fixed_e1=fixed_e1, fixed_e2=fixed_e2)
                x_algo1 = diffusion.sample(decoder, xT, norms, num_gals, method="algo1", fixed_mask_rand=fixed_mask_rand, fixed_e1=fixed_e1, fixed_e2=fixed_e2)
                x_algo2 = diffusion.sample(decoder, xT, norms, num_gals, method="algo2", fixed_mask_rand=fixed_mask_rand, fixed_e1=fixed_e1, fixed_e2=fixed_e2)

                # MSEs
                mse_direct = torch.mean((x_direct - x0) ** 2).item()
                mse_algo1  = torch.mean((x_algo1 - x0) ** 2).item()
                mse_algo2  = torch.mean((x_algo2 - x0) ** 2).item()
                mse_xT     = torch.mean((xT - x0) ** 2).item()

            # First channel only
            imgs = [
                x0.cpu().numpy()[0, 0],
                xT.cpu().numpy()[0, 0],
                x_direct.cpu().numpy()[0, 0],
                x_algo1.cpu().numpy()[0, 0],
                x_algo2.cpu().numpy()[0, 0],
            ]

            titles = [
                r"$x_0$",
                f"$x_T$\nMSE={mse_xT:.5f}",
                f"Direct\nMSE={mse_direct:.5f}",
                f"Algo 1\nMSE={mse_algo1:.5f}",
                f"Algo 2\nMSE={mse_algo2:.5f}",
            ]

            fig, axes = plt.subplots(1, 5, figsize=(20, 4))

            vmin = 0.0
            vmax = 1.0

            for ax, img, title in zip(axes, imgs, titles):
                ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_title(title)
                ax.axis("off")

            plt.tight_layout()

            filepath = os.path.join(output_dir, f"{prefix}_sample_{i+1}.png")
            plt.savefig(filepath, dpi=200)
            plt.close()

            print(f"Saved {filepath}")

    end = time.time()
    total_plots = len(datasets) * n_samples
    print(f"Plotting all {total_plots} reconstructed images took {end-start:.1f}s.")


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

            filepath = os.path.join(output_dir, f"comparison_{prefix}_sample_{i+1}.png")
            plt.savefig(filepath, dpi=200)
            plt.close()

    end = time.time()
    total_plots = len(datasets) * n_samples
    print(f"Plotting all {total_plots} reconstructed images took {end-start:.1f}s.")