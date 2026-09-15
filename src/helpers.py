import os
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


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
    all_indices = np.arange(len(dataset))
    targets = dataset.targets

    train_indices = []
    valid_indices = []

    for targ in np.unique(targets):

        # All simulations with this target
        target_indices = all_indices[targets == targ]

        # Shuffle within this cross-section
        np.random.shuffle(target_indices)
        n_train_targ = int(frac * len(target_indices))
        train_indices.extend(target_indices[:n_train_targ])
        valid_indices.extend(target_indices[n_train_targ:])

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(dataset, valid_indices)
    
    return train_dataset, valid_dataset


# Loss Plotting
def plot_losses(train_losses, valid_losses, loss_function, output_dir):
    epochs = len(train_losses)

    plt.figure(figsize=(6,4))

    plt.plot(range(1, epochs+1), train_losses, label="Train")
    plt.plot(range(1, epochs+1), valid_losses, label="Valid")

    plt.legend(loc="best")
    plt.xlabel("Epoch")
    plt.ylabel("Degradation Loss")
    plt.title(f"Training and Validation Losses ({loss_function})")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "losses_train_valid.png"), dpi=200)
    plt.close()


# Check MSE + Pearson
def evaluate_cold_diffusion(decoder, degradation, device, valid_loader, max_batches=None, model_path="decoder_best.pt"):
    """
    Evaluate Cold Diffusion reconstruction quality (MSE & Pearson).
    """
    start = time.time()

    decoder.load_state_dict(torch.load(model_path, map_location=device))
    decoder.eval()

    identity_mse_values = []
    direct_mse_values = []

    identity_corr_values = []
    direct_corr_values = []
    
    batches = 0

    with torch.no_grad():
        for x0, norms, num_gals in valid_loader:
            x0 = x0.to(device)
            norms = norms.to(device)
            num_gals = num_gals.to(device)

            # Generate fully degraded observation at t=1000.
            xT, _ = degradation.degrade(x0, norms, num_gals)

            # Reconstructed image
            x0_direct = decoder(xT)
            
            # MSEs
            identity_mse = torch.mean((x0 - xT)**2)
            direct_mse = torch.mean((x0 - x0_direct)**2)
            
            identity_mse_values.append(identity_mse.item())
            direct_mse_values.append(direct_mse.item())

            # Pearsons
            x0_flat = x0.flatten(1)
            xT_flat = xT.flatten(1)
            x0_direct_flat = x0_direct.flatten(1)

            x0_centered = x0_flat - x0_flat.mean(dim=1, keepdim=True)
            xT_centered = xT_flat - xT_flat.mean(dim=1, keepdim=True)
            x0_direct_centered = x0_direct_flat - x0_direct_flat.mean(dim=1, keepdim=True)

            eps = 1e-8

            identity_cov = (x0_centered * xT_centered).sum(dim=1)
            direct_cov = (x0_centered * x0_direct_centered).sum(dim=1)
            
            identity_std_prod = torch.sqrt((x0_centered**2).sum(dim=1))*torch.sqrt((xT_centered**2).sum(dim=1))
            direct_std_prod = torch.sqrt((x0_centered**2).sum(dim=1))*torch.sqrt((x0_direct_centered**2).sum(dim=1))
            
            identity_corr = identity_cov / (identity_std_prod + eps)
            direct_corr = direct_cov / (direct_std_prod + eps)

            identity_corr_values.extend(identity_corr.cpu().numpy())
            direct_corr_values.extend(direct_corr.cpu().numpy())

            batches += 1
            if max_batches is not None and batches >= max_batches:
                break

    end = time.time()
    print(f"\n=== Cold Diffusion Evaluation Summary ===")

    print("\n1. Baseline (Worst-Case): xT vs. x0")
    print(f"MSE     : {np.mean(identity_mse_values):.6f}")
    print(f"Pearson : {np.mean(identity_corr_values):.4f}")

    print("\n2. Direct Prediction: xT -> hat[x0]")
    print(f"MSE     : {np.mean(direct_mse_values):.6f}")
    print(f"Pearson : {np.mean(direct_corr_values):.4f}")

    print(f"MSE + Pearson Evaluation took {end-start:.1f}s.\n")


def generate_and_plot(decoder, degradation, device, train_dataset, valid_dataset, n_samples=5, model_path=None, output_dir=None):
    """
    Plot:
        True x0 | xT | Direct
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
                xT, _ = degradation.degrade(x0, norms, num_gals)

                # Reconstruction
                x_direct = decoder(xT)

            # First channel only
            imgs = [
                x0.cpu().numpy()[0, 0],
                xT.cpu().numpy()[0, 0],
                x_direct.cpu().numpy()[0, 0]
            ]

            titles = [
                r"$x_0$ (true)",
                r"$x_T$ (noised)",
                r"$\hat{x_0}$ (reconstructed)"
            ]

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

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

    end = time.time()
    total_plots = len(datasets) * n_samples
    print(f"Plotting all {total_plots} reconstructed images took {end-start:.1f}s.")


def shear_to_mass(shear_map, pad_size=100):
    """
    Converts a batch of shear maps to mass maps (convergence kappa)
    using the Kaiser-Squires inversion method.
    
    Args:
        shear_map: Tensor of shape (B, 2, H, W) where channel 0 is g1 and 1 is g2.
        pad_size: Integer, amount of zero-padding to add to each side to mitigate 
                  Fourier periodic boundary artifacts.
                  
    Returns:
        kappa_map: Tensor of shape (B, 1, H, W) representing the mass map.
    """
    B, C, H, W = shear_map.shape
    assert C == 2, "Input must have exactly 2 channels (g1, g2)"
    
    # 1. Zero-pad the shear maps to avoid periodic boundary artifacts in FFT
    # Pad order is (left, right, top, bottom)
    g_padded = F.pad(shear_map, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0.0)
    
    B_p, C_p, H_p, W_p = g_padded.shape
    g1 = g_padded[:, 0, :, :]
    g2 = g_padded[:, 1, :, :]
    
    # 2. Compute 2D FFT of the padded shear maps
    g1_hat = torch.fft.fftn(g1, dim=(-2, -1))
    g2_hat = torch.fft.fftn(g2, dim=(-2, -1))
    
    # 3. Create the Fourier space frequency grid (k_x, k_y)
    kx = torch.fft.fftfreq(W_p, device=shear_map.device)
    ky = torch.fft.fftfreq(H_p, device=shear_map.device)
    
    # Create a meshgrid for kx and ky ('ij' indexing prevents axis swapping)
    ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing='ij')
    
    k2 = kx_grid**2 + ky_grid**2
    
    # Avoid division by zero at the DC component (k=0)
    k2[0, 0] = 1.0  
    
    # 4. Compute the Kaiser-Squires Fourier space filters
    D1 = (kx_grid**2 - ky_grid**2) / k2
    D2 = (2.0 * kx_grid * ky_grid) / k2
    
    # Set the mean mass (k=0) to 0
    D1[0, 0] = 0.0
    D2[0, 0] = 0.0
    
    # 5. Apply the inversion relation: kappa_hat = D1 * g1_hat + D2 * g2_hat
    kappa_hat = D1.unsqueeze(0) * g1_hat + D2.unsqueeze(0) * g2_hat
    
    # 6. Inverse FFT to get back to spatial domain (we only take the real part)
    kappa_padded = torch.fft.ifftn(kappa_hat, dim=(-2, -1)).real
    
    # 7. Crop back to the original size
    kappa = kappa_padded[:, pad_size:-pad_size, pad_size:-pad_size]
    
    # Add the single channel dimension back (B, 1, H, W)
    return kappa.unsqueeze(1)