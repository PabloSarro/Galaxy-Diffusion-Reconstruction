import os, sys, unittest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "postmortem"))
import latent_decoder_sigma022_3_3 as ld  # noqa: E402


class TestLatentDecoder(unittest.TestCase):
    def test_decoder_predicts_noise_shape(self):
        torch.manual_seed(0)
        dec = ld.LAT.Decoder(timesteps=1000, latent_dim=128).eval()
        diff = ld.DIFF.Diffusion(timesteps=1000, device="cpu")
        x0 = torch.rand(2, 2, 32, 32)
        t = diff.sample_timesteps(2)
        xt, noise = diff.q_sample(x0, t)
        with torch.no_grad():
            out = dec(xt, torch.randn(2, 128), t)
        self.assertEqual(out.shape, noise.shape)


if __name__ == "__main__":
    unittest.main()
