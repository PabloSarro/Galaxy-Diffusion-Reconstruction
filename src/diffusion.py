import torch


class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device="cuda"):
        self.device = device
        self.timesteps = timesteps

        # Linear noise schedule
        self.beta = torch.linspace(
            beta_start,
            beta_end,
            timesteps,
            device=device
        )
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def sample_timesteps(self, batch_size):
        """
        Random timestep for each image.
        """
        return torch.randint(
            low=0,
            high=self.timesteps,
            size=(batch_size,),
            device=self.device
        )

    def q_sample(self, x0, t):
        """
        Forward diffusion.

        x0 : clean image (TS)

        Returns
        -------
        xt : noisy image
        noise : Gaussian noise used
        """

        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        xt = torch.sqrt(alpha_bar_t)*x0 + torch.sqrt(1.0-alpha_bar_t)*noise

        return xt, noise
    
    @torch.no_grad()
    def sample(self, decoder, z_obs, shape):
        """
        Reverse diffusion sampling.

        Starts from Gaussian noise and iteratively denoises.
        """

        x = torch.randn(shape, device=self.device)

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full(
                (shape[0],),
                t,
                device=self.device,
                dtype=torch.long
            )

            # Predict noise
            predicted_noise = decoder(x, z_obs, t_batch)
            alpha = self.alpha[t]
            alpha_bar = self.alpha_bar[t]
            beta = self.beta[t]

            # DDPM reverse step
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (x-((1-alpha) / torch.sqrt(1-alpha_bar))*predicted_noise) / torch.sqrt(alpha) + torch.sqrt(beta)*noise

        return x