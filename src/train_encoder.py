import time
import torch
from helpers import generate_train_valid_datasets
from torch.utils.data import DataLoader

from dataset import GalaxyDataset
from model import Encoder, contrastive_loss

from helpers import plot_losses, plot_results


NOISE_STD = 1e-2
SPARSITY = 0.25
LATENT_DIM = 128
EPOCHS = 500


# Dataset
dataset = GalaxyDataset(
    data_folder="/scratch/izar/sarro/BAHAMAS-data-nonsparse/",
    noise_std=NOISE_STD, # =================================== LOOK AT ETHAN'S CODE AND CHECK THAT NOISE_STD IS BEING ADDED IN AN EQUIVALENT WAY!! ===================================
    sparsity=SPARSITY
)

train_dataset, valid_dataset = generate_train_valid_datasets(dataset, frac=0.8)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    pin_memory=True
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=64,
    shuffle=False,
    pin_memory=True
)

# Model
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

encoder = Encoder(latent_dim=LATENT_DIM).to(device)

optimizer = torch.optim.Adam(
    encoder.parameters(),
    lr=1e-4
)

# Training
train_losses = []
valid_losses = []
best_loss = float('inf')

start = time.time()

for epoch in range(EPOCHS):
    total_train_loss = 0

    for TS, NS in train_loader:
        TS = TS.to(device, non_blocking=True)
        NS = NS.to(device, non_blocking=True)

        optimizer.zero_grad()

        z_true = encoder(TS)
        z_obs = encoder(NS)

        loss = contrastive_loss(z_true, z_obs)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    epoch_train_loss = total_train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    
    # COMPUTE VALIDATION LOSS
    encoder.eval()
    total_valid_loss = 0

    with torch.no_grad():

        for TS, NS in valid_loader:
            TS = TS.to(device, non_blocking=True)
            NS = NS.to(device, non_blocking=True)

            z_true = encoder(TS)
            z_obs = encoder(NS)

            loss = contrastive_loss(z_true, z_obs)

            total_valid_loss += loss.item()

    epoch_valid_loss = total_valid_loss / len(valid_loader)
    valid_losses.append(epoch_valid_loss)

    encoder.train()

    print(
        f"Epoch {epoch+1}/{EPOCHS} --> Train: {epoch_train_loss:.4f} | Valid: {epoch_valid_loss:.4f} (in {(time.time() - start):.2f}s)."
    )

    if epoch_valid_loss < best_loss:
        best_loss = epoch_valid_loss
        torch.save(
            encoder.state_dict(),
            "encoder_best.pt"
        )
        print("   Best model was stored")


# Visualise plot of the training and validation losses after each epoch.
plot_losses(EPOCHS, train_losses, valid_losses)

# Visualise similarity matrix between embeddings of 10 random validation simulations.
plot_results(encoder, device, valid_dataset, "encoder_best.pt")