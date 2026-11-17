import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from degradation import Degradation  # noqa: E402
from helpers import generate_train_valid_datasets, set_seed  # noqa: E402
from model import Decoder, decoder_loss  # noqa: E402


def _norm_batch(B=4, H=100, W=100, seed=0):
    """Random normalised shears in [0,1] with norms (a, b) and some empty pixels."""
    g = torch.Generator().manual_seed(seed)
    x0 = torch.rand((B, 2, H, W), generator=g) * 0.5 + 0.25  # keep |g| well below 1
    norms = torch.tensor([[-1.0, 1.0]] * B)
    num_gals = torch.ones((B, H, W))
    num_gals[:, :5] = 0  # no galaxies -> no noise there
    return x0, norms, num_gals



import unittest  # noqa: E402


class TestDegradation(unittest.TestCase):
    def test_removed_fraction_matches_sparsity(self):
        torch.manual_seed(1)
        x0, norms, ng = _norm_batch(B=8)
        _, masks = Degradation(sparsity=0.70, noise_std=0.0).degrade(x0, norms, ng)
        valid_frac = masks.float().mean().item()
        self.assertAlmostEqual(valid_frac, 0.30, delta=0.02)

    def test_sigma_zero_keeps_known_pixels(self):
        torch.manual_seed(2)
        x0, norms, ng = _norm_batch()
        xT, masks = Degradation(sparsity=0.70, noise_std=0.0).degrade(x0, norms, ng)
        # With sigma=0 and |g|<=1 the observed shear equals the true shear on kept pixels.
        g = (x0[:, 0] + 1j * x0[:, 1]) * 2 - (1 + 1j)  # de-normalise with a=-1, b=1
        keep = masks & (ng > 0) & (g.abs() <= 1)
        self.assertTrue(keep.any())
        diff = (xT - x0).abs().amax(dim=1)
        self.assertLess(diff[keep].max().item(), 1e-5)

    def test_noise_changes_known_pixels(self):
        torch.manual_seed(3)
        x0, norms, ng = _norm_batch()
        xT0, _ = Degradation(sparsity=0.70, noise_std=0.0).degrade(x0, norms, ng)
        torch.manual_seed(3)
        xT1, _ = Degradation(sparsity=0.70, noise_std=0.2).degrade(x0, norms, ng)
        self.assertGreater((xT1 - xT0).abs().max().item(), 1e-3)


class TestSplit(unittest.TestCase):
    class _Fake:
        def __init__(self):
            self.targets = np.repeat(np.arange(7), 100)

        def __len__(self):
            return len(self.targets)

    def test_split_is_deterministic_and_stratified(self):
        ds = self._Fake()
        set_seed(42)
        tr1, va1 = generate_train_valid_datasets(ds, frac=0.8)
        set_seed(42)
        tr2, va2 = generate_train_valid_datasets(ds, frac=0.8)
        self.assertEqual(list(va1.indices), list(va2.indices))
        self.assertEqual(len(tr1) + len(va1), len(ds))
        self.assertEqual(len(set(tr1.indices) & set(va1.indices)), 0)
        for t in range(7):
            n_valid = sum(ds.targets[i] == t for i in va1.indices)
            self.assertEqual(n_valid, 20)  # 80/20 split per class


class TestLoss(unittest.TestCase):
    def test_identical_inputs_give_zero_mse(self):
        x = torch.rand(2, 2, 16, 16)
        self.assertEqual(decoder_loss(x, x, method="MSE").item(), 0.0)

    def test_mse_value(self):
        a = torch.zeros(1, 2, 4, 4)
        b = torch.ones(1, 2, 4, 4)
        self.assertAlmostEqual(decoder_loss(a, b, method="MSE").item(), 1.0)

    def test_all_methods_finite_and_differentiable(self):
        for m in ["MSE", "MSE_MAE", "MSE_Grad", "Charbonnier", "Weighted"]:
            p = torch.rand(2, 2, 16, 16, requires_grad=True)
            t = torch.rand(2, 2, 16, 16)
            loss = decoder_loss(p, t, method=m, alpha=1.0)
            self.assertEqual(loss.dim(), 0, m)
            self.assertTrue(torch.isfinite(loss).item(), m)
            loss.backward()
            self.assertTrue(torch.isfinite(p.grad).all().item(), m)


class TestDecoder(unittest.TestCase):
    def test_output_shape(self):
        torch.manual_seed(0)
        net = Decoder().eval()
        with torch.no_grad():
            out = net(torch.rand(2, 2, 100, 100))
        self.assertEqual(tuple(out.shape), (2, 2, 100, 100))


if __name__ == "__main__":
    unittest.main()
