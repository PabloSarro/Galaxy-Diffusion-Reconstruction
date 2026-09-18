import os
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import griddata

# Import your own dataset classes
import netloader
from dataset import GalaxyDataset
from helpers import generate_train_valid_datasets, set_seed
from model import Decoder
from degradation import Degradation


class CustomDataset(Dataset):
    """Matches netloader's exact contract: returns (ID, target, shear_map)."""
    def __init__(self, ids, targets, shears):
        self.ids = ids
        self.targets = targets
        self.shears = shears

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.targets[idx], self.shears[idx]


def knn_reconstruct_batch(x_noised, masks):
    """
    Reconstructs missing pixels using Nearest Neighbor interpolation via scipy.griddata,
    matching the paper's baseline methodology.
    
    Args:
        x_noised: NumPy array of shape (B, C, H, W) containing the degraded images.
        masks: Boolean NumPy array of shape (B, H, W) where True indicates a valid/known pixel.
    """
    B, C, H, W = x_noised.shape
    reconstructed = np.zeros_like(x_noised)
    grid_x, grid_y = np.mgrid[0:H, 0:W]
    
    for i in range(B):
        mask = masks[i]
        # Safety check: if the image is entirely empty or entirely full, skip interpolation
        if not mask.any() or mask.all():
            reconstructed[i] = x_noised[i]
            continue
            
        # Get coordinates of the known pixels
        points = np.argwhere(mask)
        # Interpolate channel by channel (g1 and g2)
        for c in range(C):
            values = x_noised[i, c, mask]
            recon_c = griddata(points, values, (grid_x, grid_y), method='nearest')
            reconstructed[i, c] = recon_c
            
    return reconstructed


def chi_transform(red_shear):
    """
    Transforms noise-free reduced shear maps to chi maps.
    Assumes input shape is (N, C, H, W) where C is the channel dimension.
    """
    return 2 * red_shear / (1 + (red_shear ** 2).sum(axis=1, keepdims=True))


def evaluate_and_plot(preds_dict, title, output_dir):
    """Runs inference, calculates metrics, and plots the confusion matrix."""
    print(f"\nEvaluating: {title}...")
    
    # 1. Round targets to fix floating point drift (e.g., 0.099999 -> 0.1)
    true_labels = np.round(preds_dict['targets']).astype(int)
    pred_labels = np.round(np.argmax(preds_dict['preds'], axis=1)).astype(int)
    print(f"\nTrue Labels: {true_labels[235:245]}")
    print(f"Predicted Labels: {pred_labels[235:245]}")
    print(f"Unique target values: {np.unique(true_labels)}")

    # ==========================
    #    7x7 CONFUSION MATRIX
    # ==========================

    # # Sanity print to verify all 4 classes (0, 1, 2, 3) are present
    # unique_t, counts_t = np.unique(true_labels, return_counts=True)
    # print(f"  Received {len(true_labels)} samples for evaluation.")
    # for u, c in zip(unique_t, counts_t):
    #     print(f"    True Label {u}: {c} samples")

    # 2. Compute Metrics (macro average treats all classes equally)
    acc = accuracy_score(true_labels, pred_labels)
    prec, rec, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro', zero_division=0
    )
    
    # 3. Create Confusion Matrix (Forcing a 7x7 grid since the model has 7 classes)
    cm = confusion_matrix(true_labels, pred_labels, labels=np.arange(7), normalize='true')
    
    # 4. Plotting
    fig, ax = plt.subplots(figsize=(7,8))
    CLASS_NAMES = ["DARKSKIES-0", "BAHAMAS-0", "DARKSKIES-0.1", "BAHAMAS-0.1", 
                   "DARKSKIES-0.2", "BAHAMAS-0.3", "BAHAMAS-1"]
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax, cbar=False, 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel('Predictions', fontsize=12)
    ax.set_ylabel('Targets', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add metrics text box below the plot
    metrics_text = (f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1 Score: {f1:.3f}")
    plt.figtext(0.4, -0.08, metrics_text, ha='center', fontsize=12)
    
    # Save the figure
    filepath = os.path.join(output_dir, f"{title}.png")
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"\n  -> Saved plot: {title}.png")
    print(f"  -> Metrics: Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")


start_total = time.time()

# ==========================================
# 1. SET PATHS & LOCAL VARIABLES
# ==========================================
CLASSIFIER_PATH = "../results/network_v8_1.pth"
DECODER_PATH = "../results/7-with_shape_noise/MSE/training/decoder_best.pt"
DATA_FOLDER = "../data-full/"
OUTPUT_DIR = "../results/classification/plots_model_7/"

SPARSITY = 0.70
NOISE_STDS = [0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.22, 0.3]  # Noise levels for degradation
BATCH_SIZE = 32
N_CLASS = -1
# How to set N_CLASS:
# -2: Use full dataset (1200 per DARKSKIES class, 3600 per BAHAMAS class)
# -1: Use full validation split (240 per DARKSKIES class, 720 per BAHAMAS class)
# Any positive integer (e.g. 100) for a subset of the validation split (e.g. 100 per class)

TARGET_TO_CLASS = {
    0: "DARKSKIES-0",
    1: "BAHAMAS-0",
    2: "DARKSKIES-0.1",
    3: "BAHAMAS-0.1",
    4: "DARKSKIES-0.2",
    5: "BAHAMAS-0.3",
    6: "BAHAMAS-1"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. LOAD DATA & MODELS
# ==========================================
print(f"Loading classifier from: {CLASSIFIER_PATH}")
start = time.time()
arch = torch.load(CLASSIFIER_PATH, map_location=device, weights_only=False)
print(f"    Done ({time.time() - start:.2f}s)")

print(f"Loading U-Net Decoder from: {DECODER_PATH}")
start = time.time()
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
decoder.eval()
print(f"    Done ({time.time() - start:.2f}s)")

print(f"Loading degradation with: sp={SPARSITY}, stds in [{NOISE_STDS}]")
start = time.time()
degradations = []
for NOISE_STD in NOISE_STDS:
    degradation = Degradation(sparsity=SPARSITY, noise_std=NOISE_STD, device=device)
    degradations.append(degradation)
print(f"    Done ({time.time() - start:.2f}s)")

print(f"Loading data from: {DATA_FOLDER}")
start = time.time()
dataset = GalaxyDataset(data_folder=DATA_FOLDER)
if N_CLASS != -2:
    _, valid_dataset = generate_train_valid_datasets(dataset, frac=0.8)
    indices_array = np.array(valid_dataset.indices)
else:
    # dataset doesn't have indices attribute:
    indices_array = np.arange(len(dataset))

targets = dataset.targets[indices_array]

# Extract exactly N indices for each unique target
if N_CLASS == -2:
    print("    Using the full dataset: 1200/3600 samples per class with DARKSKIES/BAHAMAS.")
    indices = indices_array
elif N_CLASS == -1:
    print("    Using the full validation split: 240/720 samples per class with DARKSKIES/BAHAMAS.")
    indices = indices_array
else:
    print(f"    Using a balanced subset: {N_CLASS} validation samples per class.")
    subset_indices = []
    for targ in np.unique(dataset.targets):
        matching_indices = indices_array[targets == targ]
        if len(matching_indices) < N_CLASS:
            print(f"Warning: Only {len(matching_indices)} samples available for target={targ}")
            subset_indices.extend(matching_indices)
        else:
            subset_indices.extend(matching_indices[:N_CLASS])

    indices = np.array(subset_indices)

subset = torch.utils.data.Subset(dataset, indices)
loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# Un-normalise to get original reduced shear values.
base_images_norm = dataset.images[indices]
base_norms = dataset.norms[indices]
a = base_norms[:, 0, np.newaxis, np.newaxis, np.newaxis]
b = base_norms[:, 1, np.newaxis, np.newaxis, np.newaxis]
base_images = base_images_norm * (b-a) + a

# Apply the non-linear transformation (model takes images as chi maps).
base_images = chi_transform(base_images)

base_targets = dataset.targets[indices]
sample_ids = np.arange(len(base_targets))

print(f"    Done ({time.time() - start:.2f}s)")

# # SANITY CHECK: Print number of samples per class.
# all_vals, all_counts = np.unique(dataset.targets, return_counts=True)
# print("\nDataset Label Counts")
# print("  Full Dataset:")
# for val, count in zip(all_vals, all_counts):
#     print(f"    Class {TARGET_TO_CLASS[val]} (target {val}): {count} total (Expected 1200/3600)")

# valid_vals, valid_counts = np.unique(val_targets, return_counts=True)
# print("\n  Evaluation Dataset:")
# for val, count in zip(valid_vals, valid_counts):
#     print(f"    Class {TARGET_TO_CLASS[val]} (target {val}): {count} samples (Expected 240/720)")

# ===============================
# 3. GENERATE RECONSTRUCTIONS
# ===============================
print("Generating Noisy, U-Net, and kNN datasets...")
start = time.time()

noisy_images_list = []
unet_images_list = [[] for _ in range(len(degradations))]
knn_images_list = []

with torch.no_grad():
    for x0, norms, num_gals in loader:
        x0 = x0.to(device, non_blocking=True)
        norms = norms.to(device, non_blocking=True)
        num_gals = num_gals.to(device, non_blocking=True)

        # 1. Noisy images
        x_noised_0, masks_0 = degradations[0].degrade(x0, norms, num_gals)
        noisy_images_list.append(x_noised_0.cpu().numpy())       
        
        for idx, degradation in enumerate(degradations):
            # 2. U-Net Reconstruct for every degradation
            x_noised, masks = degradation.degrade(x0, norms, num_gals)
            x0_pred = decoder(x_noised)
            unet_images_list[idx].append(x0_pred.cpu().numpy())
        
        # 3. kNN Reconstruct
        x_knn = knn_reconstruct_batch(x_noised_0.cpu().numpy(), masks_0.cpu().numpy())
        knn_images_list.append(x_knn)

# Concatenate lists into large numpy arrays
noisy_images = np.concatenate(noisy_images_list, axis=0)
for idx in range(len(degradations)):
    unet_images_list[idx] = np.concatenate(unet_images_list[idx], axis=0)
knn_images = np.concatenate(knn_images_list, axis=0)

# Un-normalise
noisy_images = noisy_images * (b-a) + a
for idx in range(len(degradations)):
    unet_images_list[idx] = unet_images_list[idx] * (b-a) + a
knn_images = knn_images * (b-a) + a

noisy_images = chi_transform(noisy_images)
for idx in range(len(degradations)):
    unet_images_list[idx] = chi_transform(unet_images_list[idx])
knn_images = chi_transform(knn_images)

image_sets = {
    "1. Noise-Free": base_images,
    "2. Noisy": noisy_images,
    **{f"3. U-Net-degr{NOISE_STDS[idx]}": unet_images_list[idx] for idx in range(len(NOISE_STDS))},
    "4. kNN": knn_images
}

print(f"    Done ({time.time() - start:.2f}s)")

# ===========================
# 4. RUN INFERENCE PIPELINE
# ===========================
all_predictions = {}

print(f"\n======== Evaluation loop starting... (Results saving to {OUTPUT_DIR}) ========")
for set_name, raw_shears in image_sets.items():
    print(f"\n------ Predicting: {set_name} ------")
    start = time.time()
    
    # 1. Instantiate CustomDataset
    test_dataset = CustomDataset(
        ids=sample_ids,
        targets=base_targets.copy(),
        shears=raw_shears.copy()
    )
    
    # 2. Transform shears and targets using the network's built-in transforms
    test_dataset.shears = arch.transforms['inputs'](test_dataset.shears)
    test_dataset.targets = arch.transforms['targets'](test_dataset.targets)

    # 3. Create test data loader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Generate predictions
    preds = arch.predict(test_loader)
    all_predictions[set_name] = preds

    evaluate_and_plot(preds, set_name, OUTPUT_DIR)
    print(f"  Done ({time.time() - start:.2f}s)")

        
print(f"\nTotal evaluation time: {time.time() - start_total:.2f}s")