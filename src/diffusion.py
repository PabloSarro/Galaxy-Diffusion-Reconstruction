import torch
import numpy as np
import torch.nn.functional as F

class Diffusion:

    def __init__(self, timesteps=1000, max_sparsity=0.25, max_noise_std=0.01, device="cuda"):
        self.device = device
        self.timesteps = timesteps
        self.max_sparsity = max_sparsity
        self.max_noise_std = max_noise_std


    def sample_timesteps(self, batch_size):
        """
        Random timestep for each image.
        """
        return torch.randint(
            low=1,
            high=self.timesteps+1, # +1, since randint(low, high) does: [low, high)
            size=(batch_size,),
            device=self.device
        )
    

    def degrade(self, x0, norms, num_gals, t, fixed_mask_rand=None, fixed_e1=None, fixed_e2=None):
        """
        Cold Diffusion forward degradation operator: D(x0, t).
        Scales sparcity and noise_std linearly by timestep t / T.

        fixed_e1/e2: 
            These need to be fixed during inference, since x_{t-1} = x_t - \hat{x}_t + \hat{x}_{t-1},
            where \hat{} represents running the degradation function D. We need both evaluations to
            have the same random intrinsic galaxy ellipticity noise, since we want to get a single
            galaxy shape at the end of the inference process.

        Output: x_t = D(x0, t), the degraded image
        """

        B, C, H, W = x0.shape
        device = x0.device

        # Scale parameters by timestep fraction
        t_norm = t.float() / self.timesteps  # (B,)
        t_norm_3d = t_norm.view(B, 1, 1)     # (B,) --> (B, H, W) (extend t's in a batch so that every pixel in an image has the same timestep, for all images in the batch).
        # t_norm_4d = t_norm.view(B, 1, 1, 1)  # (B,) --> (B, C, H, W) (keep it one-dimensional, since this is a pixel-wise operation, so every pixel requires a single float)

        sparsity_t = t_norm_3d * self.max_sparsity
        noise_std_t = t_norm_3d * self.max_noise_std

        xt = x0.clone()
        ng = num_gals.clone()

        a = norms[:, 0].view(B, 1, 1)
        b = norms[:, 1].view(B, 1, 1)

        # ===================================================
        # ================ 1. Random masking ================
        # ===================================================

        # If we give the degradation process a fixed mask, apply that one.
        if fixed_mask_rand is not None:
            mask_rand = fixed_mask_rand
        # Else, make it random.
        else:
            mask_rand = torch.rand((B, H, W), device=device)

        # Mask of removed pixels.
        removed = (mask_rand < sparsity_t)
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
        sigma_pixel = noise_std_t / np.sqrt(N_gal) # if isinstance(noise_std_t, torch.Tensor) else noise_std_t / np.sqrt(N_gal)

        # If we give the degradation process a fixed noise schedule, assign that one.
        if (fixed_e1 is not None) and (fixed_e2 is not None):
            e1 = fixed_e1 * sigma_pixel
            e2 = fixed_e2 * sigma_pixel
        # Otherwise, make it random.
        else:
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


    @torch.no_grad()
    def sample(self, decoder, x, norms, num_gals, method="algo2", fixed_mask_rand=None, fixed_e1=None, fixed_e2=None):
        """
        Cold Diffusion reverse sampling operator: R(xt, t)
        This starts from the observed x (at t = T).

        method:
            "direct" -> xT -> x0 prediction only
            "algo1"  -> Algorithm 1
            "algo2"   -> Algorithm 2
        
        Output: x0 = R(xt, t), the predicted clean image.
        """

        B, C, H, W = x.shape
        device = self.device

        # Pre-sample consistent randomness for the backward trajectory steps
        if fixed_mask_rand is None:
            fixed_mask_rand = torch.rand((B, H, W), device=device)
        if fixed_e1 is None:
            fixed_e1 = torch.randn((B, H, W), device=device)
        if fixed_e2 is None:
            fixed_e2 = torch.randn((B, H, W), device=device)

        # ==========================================================
        # ============ 1. Direct prediction: xT -> x0 ==============
        # ==========================================================
        if method == "direct":
            t_max = torch.full((B,), self.timesteps, device=device, dtype=torch.long)
            return decoder(x, t_max)

        # ==========================================================
        # ============= 2. Naive algorithm: xT -> x0 ===============
        # ==========================================================

        # Idea: xt --[R]-> x0_pred --[D]-> x_{t-1} --[R]-> x0_pred --[D]-> x_{t-2} --> ... --> x_1 --[R]-> x_0.

        elif method == "algo1":
            for t in reversed(range(1, self.timesteps + 1)): # for t in timesteps+1, timesteps, timesteps-1, ..., 4, 3, 2, 1.
                t_batch = torch.full((B,), t, device=device, dtype=torch.long)
                t_prev_batch = torch.full((B,), t - 1, device=device, dtype=torch.long)

                # Predict clean image x0 directly
                x0_pred = decoder(x, t_batch) # \hat{x0}
                
                x = self.degrade(x0_pred, norms, num_gals, t_prev_batch, fixed_mask_rand=fixed_mask_rand, fixed_e1=fixed_e1, fixed_e2=fixed_e2)
            
            return x

        # ==========================================================
        # =========== 3. Complete algorithm: xT -> x0 ==============
        # ==========================================================

        # Idea: xt --[R]-> x0_pred --> x_{t-1} = xt-D(x0_pred, t)+D(x0_pred, t-1) --[R]-> x0_pred --> ... --> x_1 = x_2-D(x0_pred, 2)+D(x0_pred, 1) --[R]-> x0_pred --> x0 = x1-D(x0_pred, 1)+D(x0_pred, 0)=x1-D(x0_pred)+x0_pred.

        elif method =="algo2":
            for t in reversed(range(1, self.timesteps + 1)): # for t in timesteps+1, timesteps, timesteps-1, ..., 4, 3, 2, 1.
                t_batch = torch.full((B,), t, device=device, dtype=torch.long)
                t_prev_batch = torch.full((B,), t - 1, device=device, dtype=torch.long)

                # Predict clean image x0 directly
                x0_pred = decoder(x, t_batch) # \hat{x0}
                
                xt_hat = self.degrade(x0_pred, norms, num_gals, t_batch, fixed_mask_rand=fixed_mask_rand, fixed_e1=fixed_e1, fixed_e2=fixed_e2)
                xt_prev_hat = self.degrade(x0_pred, norms, num_gals, t_prev_batch, fixed_mask_rand=fixed_mask_rand, fixed_e1=fixed_e1, fixed_e2=fixed_e2)
                
                x = x - xt_hat + xt_prev_hat

            return x