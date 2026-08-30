import os
import glob
import torch
import pickle
import numpy as np

from torch.utils.data import Dataset


class GalaxyDataset(Dataset):

    def __init__(self, data_folder):
        """
        Parameters
        ----------
        data_folder : str
            Path to the folder containing all the .pkl simulation files.
        """
        
        self.images = []
        self.norms = []
        self.num_gals = []
        self.cross_sections = []
        self.targets = []

        files = glob.glob(
            os.path.join(data_folder, "*.pkl")
        )

        for file in files:
            with open(file, "rb") as f:
                metadata, images = pickle.load(f)

            # Store pixel images (x0).
            images = images.astype(np.float32)
            self.images.append(images)

            # Store min/max of every observation.
            norms = metadata["norms"].astype(np.float32)
            self.norms.append(norms)

            # Store number of galaxies per pixel.
            num_gals = metadata["num_gals"].astype(np.float32) # Stored to know where to apply noise --> only in informative pixels! There is no noise where there are no galaxies behind!
            self.num_gals.append(num_gals)

            # Store cross-section per image.
            n_samples = len(images)
            sigma = metadata["label"][0]
            self.cross_sections.extend([sigma] * n_samples)

            # Store cross-section per image.
            filename = os.path.basename(file).lower()
            if "darkskies" in filename:
                if np.isclose(sigma, 0.0, atol=1e-2) or np.isclose(sigma, 0.01, atol=1e-2):
                    class_idx = 0
                elif np.isclose(sigma, 0.1, atol=1e-2):
                    class_idx = 2
                elif np.isclose(sigma, 0.2, atol=1e-2):
                    class_idx = 4
                else:
                    raise ValueError(f"Unknown DARKSKIES sigma: {sigma} in {filename}")
                    
            elif "bahamas" in filename:
                if np.isclose(sigma, 0.0, atol=1e-2) or np.isclose(sigma, 0.01, atol=1e-2):
                    class_idx = 1
                elif np.isclose(sigma, 0.1, atol=1e-2):
                    class_idx = 3
                elif np.isclose(sigma, 0.3, atol=1e-2):
                    class_idx = 5
                elif np.isclose(sigma, 1.0, atol=1e-2):
                    class_idx = 6
                else:
                    raise ValueError(f"Unknown BAHAMAS sigma: {sigma} in {filename}")

            self.targets.extend([class_idx] * n_samples)

        # Merge all files
        self.images = np.concatenate(self.images, axis=0)
        self.norms = np.concatenate(self.norms, axis=0)
        self.num_gals = np.concatenate(self.num_gals, axis=0)
        self.cross_sections = np.array(self.cross_sections)
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # We only return the clean data (x0) and metadata.
        # The noise will be added dynamically by the Diffusion class based on the timestep 't'.
        x0 = self.images[idx]
        norms = self.norms[idx]
        num_gals = self.num_gals[idx]

        x0 = torch.tensor(x0, dtype=torch.float32)
        norms = torch.tensor(norms, dtype=torch.float32)
        num_gals = torch.tensor(num_gals, dtype=torch.float32)

        return x0, norms, num_gals
   
    def get_cross_section(self, idx):
        return self.cross_sections[idx]

    def get_target(self, idx):
        return self.targets[idx]