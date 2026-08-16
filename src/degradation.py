import torch
import numpy as np

class Degradation:

    def __init__(self, sparsity=0.25, noise_std=0.01, device="cuda"):
        self.device = device
        self.sparsity = sparsity
        self.noise_std = noise_std
    

    def degrade(self, x0, norms, num_gals):
        """
        Physical degradation operator.

        Applies:
        1. Random galaxy masking.
        2. Median replacement of removed pixels.
        3. Intrinsic ellipticity noise.
        4. Normalisation back to network input space.

        Output:
            xT = D(x0), degraded image.
        """

        B, C, H, W = x0.shape
        device = x0.device

        sparsity = self.sparsity
        noise_std = self.noise_std

        xt = x0.clone()
        ng = num_gals.clone()

        a = norms[:, 0].view(B, 1, 1)
        b = norms[:, 1].view(B, 1, 1)

        # ===================================================
        # ================ 1. Random masking ================
        # ===================================================

        # Find random sparsity mask.
        mask_rand = torch.rand((B, H, W), device=device)

        # Mask of removed pixels.
        removed = (mask_rand < sparsity)
        # No galaxies remain in these pixels.
        ng[removed] = 0
        # And replace the intensities of such pixels by the median intensity of the image.
        for i in range(B):
            median0 = torch.median(xt[i, 0])
            median1 = torch.median(xt[i, 1])
            xt[i, 0, removed[i]] = median0 # x_t has dim: (B, C, H, W) --> i-th image in the batch, channel 0, removed mask
            xt[i, 1, removed[i]] = median1

        # ===================================================
        # ======= 2. De-normalise to physical shears ========
        # ===================================================

        g1 = xt[:, 0]*(b-a) + a # ":" --> across all batch images ; "0" --> for channel 0
        g2 = xt[:, 1]*(b-a) + a # ":" --> across all batch images ; "1" --> for channel 1
        g = g1 + 1j*g2

        # ===================================================
        # ========== 3. Apply ellipticity noising ===========
        # ===================================================
        
        # Pixels containing galaxies
        valid_pixels = (ng > 0)
        # Intrinsic ellipticity dispersion
        N_gal = 1 # Assumption, PENDING TO VERIFY!!
        sigma_pixel = noise_std / np.sqrt(N_gal) # if isinstance(noise_std, torch.Tensor) else noise_std / np.sqrt(N_gal)

        # Random noise schedule.
        e1 = torch.randn((B, H, W), device=device) * sigma_pixel
        e2 = torch.randn((B, H, W), device=device) * sigma_pixel

        eps_s = e1 + 1j * e2
        eps_obs = g.clone()
        abs_g = torch.abs(g)

        weak = valid_pixels & (abs_g <= 1)
        strong = valid_pixels & (abs_g > 1)

        # |g| <= 1
        eps_obs[weak] = (eps_s[weak] + g[weak]) / (1 + torch.conj(g[weak])*eps_s[weak])
        # |g| > 1
        eps_obs[strong] = (1 + g[strong]*torch.conj(eps_s[strong])) / (torch.conj(eps_s[strong]) + torch.conj(g[strong]))

        g1_obs = eps_obs.real
        g2_obs = eps_obs.imag

        # ===================================================
        # ================ 4. Normalise back ================
        # ===================================================

        xt[:, 0] = (g1_obs-a) / (b-a)
        xt[:, 1] = (g2_obs-a) / (b-a)

        return xt