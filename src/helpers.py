import time
import torch
import random
import numpy as np

import matplotlib.pyplot as plt
import torch.nn.functional as F


# DETERMINISTIC RUNS FOR COMPARISON
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ENCODER & DECODER
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
    print(f"Generating the Train+Valid Datasets took {end-start} seconds.")
    
    return train_dataset, valid_dataset


# ENCODER & DECODER
def plot_losses(epochs, train_losses, valid_losses):
    plt.figure(figsize=(6,4))

    plt.plot(range(1, epochs+1), train_losses, label="Train")
    plt.plot(range(1, epochs+1), valid_losses, label="Valid")

    plt.legend(loc="best")
    plt.xlabel("Epoch")
    plt.ylabel("Contrastive Loss")
    plt.title("Contrastive Training and Validation Losses")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("train_valid_losses.png", dpi=200)
    plt.show()


# ENCODER
def plot_results(encoder, device, valid_dataset, encoder_path):

    encoder.load_state_dict(
        torch.load(encoder_path, map_location=device)
    )
    encoder.eval()
    cross_sections = valid_dataset.dataset.cross_sections

    # Randomly select 10 samples from validation set
    valid_indices = np.arange(len(valid_dataset))
    sigma_groups = {}

    for idx in valid_indices:
        sigma = cross_sections[valid_dataset.indices[idx]]

        if sigma not in sigma_groups:
            sigma_groups[sigma] = []

        sigma_groups[sigma].append(idx)

    metrics = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (sigma, indices) in enumerate(sigma_groups.items()):

        # Pick 10 random validation samples
        n_samples = min(10, len(indices))
        selected = np.random.choice(
            indices,
            size=n_samples,
            replace=False
        )

        TS_list = []
        NS_list = []

        for idx in selected:
            TS, NS = valid_dataset[idx]
            TS_list.append(TS)
            NS_list.append(NS)

        TS_batch = torch.stack(TS_list).to(device)
        NS_batch = torch.stack(NS_list).to(device)

        with torch.no_grad():

            z_true = encoder(TS_batch)
            z_obs = encoder(NS_batch)

            z_true = F.normalize(z_true, dim=1)
            z_obs = F.normalize(z_obs, dim=1)

            # 10x10 cosine similarity matrix
            similarity_matrix = torch.matmul(z_true, z_obs.T)

        similarity_matrix = similarity_matrix.cpu().numpy()

        # Metrics
        target = np.eye(n_samples)

        mse = np.mean((similarity_matrix - target)**2)
        diagonal = np.mean(np.diag(similarity_matrix))
        off_diagonal = np.mean(similarity_matrix[~np.eye(n_samples, dtype=bool)])
        separation = diagonal / (off_diagonal+1e-10)

        metrics[sigma] = {
            "MSE": mse,
            "Diagonal": diagonal,
            "Off-diagonal": off_diagonal,
            "Separation": separation
        }

        # Plot
        ax = axes[i]
        im = ax.imshow(similarity_matrix, vmin=-1, vmax=1, cmap="viridis")
        fig.colorbar(im, ax=ax, label="Cosine similarity")

        for i in range(n_samples):
            for j in range(n_samples):
                ax.text(j, i, f"{similarity_matrix[i,j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8
                )

        ax.set_xticks(range(n_samples))
        ax.set_yticks(range(n_samples))
        ax.set_xlabel("NS embedding index")
        ax.set_ylabel("TS embedding index")
        ax.set_title(f"Cross-section = {sigma:.3e}\nMSE={mse:.4e}")

    plt.tight_layout()
    plt.savefig("cosine_matrices_all_sigma.png", dpi=200)
    plt.close()

    # Print summary table
    print("\n=== Encoder performance by cross-section ===")

    for sigma, values in metrics.items():
        print(f"\nσ/m = {sigma:.3e}")
        print(f"MSE          : {values['MSE']:.6f}")
        print(f"Diagonal     : {values['Diagonal']:.4f}")
        print(f"Off-diagonal : {values['Off-diagonal']:.4f}")
        print(f"Separation   : {values['Separation']:.4f}")



# DECODER
def evaluate_reconstruction(encoder, decoder, diffusion, device, valid_loader, max_batches=None):
    """
    Evaluate diffusion reconstruction quality.

    Returns:
        mean MSE
        mean Pearson correlation
    """

    start = time.time()

    decoder.load_state_dict(
        torch.load("decoder_best.pt")
    )
    encoder.eval()
    decoder.eval()

    mse_values = []
    corr_values = []
    batches = 0

    with torch.no_grad():

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        for TS, NS in valid_loader:
            TS = TS.to(device)
            NS = NS.to(device)

            # Condition
            z_obs = encoder(NS)

            # Generate reconstruction
            reconstruction = diffusion.sample(decoder, z_obs, TS.shape)

            # MSE
            mse = torch.mean((TS - reconstruction)**2)
            mse_values.append(mse.item())

            # Pearson correlation
            TS_flat = TS.flatten(1)
            rec_flat = reconstruction.flatten(1)

            TS_centered = TS_flat - TS_flat.mean(dim=1, keepdim=True)
            rec_centered = rec_flat - rec_flat.mean(dim=1, keepdim=True)

            eps = 1e-8
            cov = (TS_centered * rec_centered).sum(dim=1)
            std_prod = torch.sqrt((TS_centered**2).sum(dim=1))*torch.sqrt((rec_centered**2).sum(dim=1))
            
            corr = cov / (std_prod + eps)
            corr_values.extend(corr.cpu().numpy())

            batches += 1
            if max_batches is not None and batches >= max_batches:
                break

    end = time.time()
    print(f"MSE: {np.mean(mse_values)} | Pearson: {np.mean(corr_values)} (this took {end-start} seconds)")



# DECODER
def plot_reconstruction_img(encoder, decoder, diffusion, device, valid_dataset):

    start = time.time()

    decoder.load_state_dict(
        torch.load("decoder_best.pt")
    )
    encoder.eval()
    decoder.eval()

    TS, NS = valid_dataset[0]

    TS = TS.unsqueeze(0).to(device)
    NS = NS.unsqueeze(0).to(device)

    with torch.no_grad():
        z_obs = encoder(NS)
        reconstruction = diffusion.sample(decoder, z_obs, TS.shape)

    TS = TS.cpu().numpy()[0]
    NS = NS.cpu().numpy()[0]
    reconstruction = reconstruction.cpu().numpy()[0]

    fig, ax = plt.subplots(1,3, figsize=(12,4))

    ax[0].imshow(TS[0])
    ax[0].set_title("TS (true)")

    ax[1].imshow(NS[0])
    ax[1].set_title("NS (observed)")

    ax[2].imshow(reconstruction[0])
    ax[2].set_title("Reconstruction")

    plt.tight_layout()
    plt.savefig("decoder_reconstruction.png", dpi=200)

    end = time.time()
    print(f"Plotting the reconstructed image took {end-start} seconds.")
