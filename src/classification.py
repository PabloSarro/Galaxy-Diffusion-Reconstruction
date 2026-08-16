import torch
from helpers import shear_to_mass

# Load Bayesian-DARKSKIES Classifier
# Iterate across ['noise-free images', 'noisy images', 'reconstructed images']
    # Take a bunch of shear maps
    # Transform them into mass maps (helper function)
    # Pass them through the classifier.
    # Compare against true labels and compute accuracy, recall, precision, F1,... across cross-sections.
    # Store in vectors and plot (maybe one plot per cross-section).