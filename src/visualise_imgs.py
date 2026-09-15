import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Import your own dataset classes
from dataset import GalaxyDataset
from model import Decoder
from degradation import Degradation
from helpers import set_seed, generate_train_valid_datasets


def knn_reconstruct_batch(x_noised, masks):
    B, C, H, W = x_noised.shape
    reconstructed = np.zeros_like(x_noised)
    grid_x, grid_y = np.mgrid[0:H, 0:W]
    
    for i in range(B):
        mask = masks[i]
        if not mask.any() or mask.all():
            reconstructed[i] = x_noised[i]
            continue
            
        points = np.argwhere(mask)
        for c in range(C):
            values = x_noised[i, c, mask]
            recon_c = griddata(points, values, (grid_x, grid_y), method='nearest')
            reconstructed[i, c] = recon_c
            
    return reconstructed

def chi_transform(red_shear):
    return 2 * red_shear / (1 + (red_shear ** 2).sum(axis=1, keepdims=True))


# 1. SET PATHS & LOCAL VARIABLES
DECODER_PATH = "../results/5-low_resolution/MSE/training/decoder_best.pt"
DATA_FOLDER = "../data-full/"
OUTPUT_DIR = "../results/classification/"

SPARSITY = 0.70
NOISE_STD = 0.005

os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. LOAD DATA & MODELS
print("Loading models and dataset...")
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
decoder.eval()

degradation = Degradation(sparsity=SPARSITY, noise_std=NOISE_STD, device=device)
dataset = GalaxyDataset(data_folder=DATA_FOLDER)

# Split the dataset
_, valid_dataset = generate_train_valid_datasets(dataset, frac=0.8)
valid_indices = np.array(valid_dataset.indices)
valid_targets = dataset.targets[valid_indices]

# Target IDs for BAHAMAS-0, BAHAMAS-0.1, BAHAMAS-0.3, BAHAMAS-1
target_classes = [1, 3, 5, 6]
class_names = ["BAHAMAS-0", "BAHAMAS-0.1", "BAHAMAS-0.3", "BAHAMAS-1"]

# Find one index for each target class
sample_indices = []
for targ in target_classes:
    idx = np.where(valid_targets == targ)[0][0]
    sample_indices.append(idx)

# 3. EXTRACT AND PROCESS SAMPLES
x0_list, norms_list, num_gals_list = [], [], []

for idx in sample_indices:
    x0, norms, num_gals = valid_dataset[idx]
    x0_list.append(x0)
    norms_list.append(norms)
    num_gals_list.append(num_gals)

# Stack into batches
batch_x0 = torch.stack(x0_list).to(device)
batch_norms = torch.stack(norms_list).to(device)
batch_num_gals = torch.stack(num_gals_list).to(device)

print("Generating reconstructions...")
with torch.no_grad():
    x_noised, masks = degradation.degrade(batch_x0, batch_norms, batch_num_gals)
    x0_pred = decoder(x_noised)
    x_knn = knn_reconstruct_batch(x_noised.cpu().numpy(), masks.cpu().numpy())

# Convert to numpy arrays
base_images_norm = batch_x0.cpu().numpy()
noisy_images = x_noised.cpu().numpy()
unet_images = x0_pred.cpu().numpy()
knn_images = x_knn

# Un-normalise
base_norms = batch_norms.cpu().numpy()
a = base_norms[:, 0, np.newaxis, np.newaxis, np.newaxis]
b = base_norms[:, 1, np.newaxis, np.newaxis, np.newaxis]

base_images = base_images_norm * (b-a) + a
noisy_images = noisy_images * (b-a) + a
unet_images = unet_images * (b-a) + a
knn_images = knn_images * (b-a) + a

# Apply chi_transform
base_images = chi_transform(base_images)
noisy_images = chi_transform(noisy_images)
unet_images = chi_transform(unet_images)
knn_images = chi_transform(knn_images)

image_sets = {
    "Noise-Free": base_images,
    "Noisy": noisy_images,
    "U-Net": unet_images,
    "kNN": knn_images
}

# 4. VISUALISATION PLOT
print("Plotting grid...")
set_keys = list(image_sets.keys())
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))

for row_idx, target_val in enumerate(target_classes):
    for col_idx, set_name in enumerate(set_keys):
        ax = axes[row_idx, col_idx]
        
        # Select channel 0 of the current image
        img = image_sets[set_name][row_idx, 0, :, :]
        
        ax.imshow(img, cmap='viridis')
        ax.axis('off')
        
        if row_idx == 0:
            ax.set_title(set_name, fontsize=14)
            
        if col_idx == 0:
            ax.text(-0.1, 0.5, class_names[row_idx], va='center', ha='right', 
                    transform=ax.transAxes, fontsize=14, rotation=90)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "visualise_imgs.png")
plt.savefig(plot_path, bbox_inches="tight", dpi=300)
plt.close()
print(f"-> Saved visualisation to {plot_path}")