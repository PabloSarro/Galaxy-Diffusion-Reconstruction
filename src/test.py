import matplotlib.pyplot as plt
from dataset import GalaxyDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = GalaxyDataset(
    data_folder="/scratch/izar/sarro/BAHAMAS-data-nonsparse/",
    noise_std=0.01,
    sparsity=0.25
)

# DataLoader
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

# Get one sample
TS, NS = next(iter(loader))

print("TS shape:", TS.shape)
print("NS shape:", NS.shape)

# Remove batch dimension
TS = TS[0].numpy()
NS = NS[0].numpy()

# Plot both channels
fig, ax = plt.subplots(
    2,
    2,
    figsize=(10, 8)
)

for c in range(2):
    # True simulation
    im1 = ax[c, 0].imshow(
        TS[c],
        origin="lower"
    )
    ax[c, 0].set_title(
        f"TS - Channel {c}"
    )
    plt.colorbar(
        im1,
        ax=ax[c,0]
    )

    # Noisy observation
    im2 = ax[c, 1].imshow(
        NS[c],
        origin="lower"
    )
    ax[c, 1].set_title(
        f"NS - Channel {c}"
    )
    plt.colorbar(
        im2,
        ax=ax[c,1]
    )

plt.tight_layout()
plt.savefig("TS_vs_NS.png", dpi=200)