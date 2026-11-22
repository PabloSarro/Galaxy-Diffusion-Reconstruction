import os, sys, unittest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "postmortem"))
import encoder_sigma022_3_3 as e  # noqa: E402


class TestEncoder(unittest.TestCase):
    def test_encoder_shape_and_finite_loss(self):
        torch.manual_seed(0)
        enc = e.LAT.Encoder(latent_dim=128)
        x = torch.rand(4, 2, 100, 100)
        z = enc(x)
        self.assertEqual(tuple(z.shape), (4, 128))
        loss = e.LAT.contrastive_loss(enc(x), enc(x + 0.01 * torch.randn_like(x)))
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
