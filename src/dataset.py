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

        # Merge all files
        self.images = np.concatenate(self.images, axis=0)
        self.norms = np.concatenate(self.norms, axis=0)
        self.num_gals = np.concatenate(self.num_gals, axis=0)
        self.cross_sections = np.array(self.cross_sections)

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



# OLD ADD_NOISE FUNCTION:
    # NS = TS.copy()
    # num_gals = num_gals.copy()

    # a, b = norms

    # # ===================================================
    # # ================ 1. Random masking ================
    # # ===================================================
    # if self.sparsity > 0:
    #     mask = np.random.rand(*num_gals.shape)
    #     removed = (mask < self.sparsity)
        
    #     # No galaxies remain in these pixels
    #     num_gals[removed] = 0

    #     # Replace by image background (median), not zero
    #     median0 = np.median(NS[0])
    #     median1 = np.median(NS[1])

    #     NS[0, removed] = median0
    #     NS[1, removed] = median1


    # # ===================================================
    # # ======= 2. De-normalise to physical shears ========
    # # ===================================================
    # g1 = NS[0]*(b-a) + a
    # g2 = NS[1]*(b-a) + a

    # g = g1 + 1j*g2


    # # ===================================================
    # # ========== 3. Apply ellipticity noising ===========
    # # ===================================================

    # # Pixels containing galaxies
    # valid_pixels = num_gals > 0

    # # Intrinsic ellipticity dispersion
    # N_gal = 1 # and N_gal=1 was assumed.
    # sigma_pixel = self.noise_std / np.sqrt(N_gal) # usually sigma_e ~ 0.3 for weak lensing.

    # # Generate galaxy shape noise
    # e1, e2 = np.random.normal(
    #     loc=0.0,
    #     scale=sigma_pixel,
    #     size=NS.shape
    # )

    # eps_s = e1 + 1j*e2
    # eps_obs = g.copy()
    # abs_g = np.abs(g)

    # weak = valid_pixels & (abs_g <= 1)
    # strong = valid_pixels & (abs_g > 1)

    # # |g| <= 1
    # eps_obs[weak] = (eps_s[weak] + g[weak]) / (1 + np.conj(g[weak])*eps_s[weak])

    # # |g| > 1
    # eps_obs[strong] = (1 + g[strong]*np.conj(eps_s[strong])) / (np.conj(eps_s[strong]) + np.conj(g[strong]))

    # g1 = eps_obs.real
    # g2 = eps_obs.imag

    # # ===================================================
    # # ================ 4. Normalise back ================
    # # ===================================================
    # NS[0] = (g1-a) / (b-a)
    # NS[1] = (g2-a) / (b-a)

    # return NS # Some shear values may be <0 or >1, but that should be fine.