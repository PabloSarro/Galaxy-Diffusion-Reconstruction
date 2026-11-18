import os, sys, unittest
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eval_split import per_sample_metrics, score_split  # noqa: E402
from degradation import Degradation  # noqa: E402


class TestEvalSplit(unittest.TestCase):
    def test_perfect_prediction(self):
        x = torch.rand(4, 2, 8, 8)
        mse, rho = per_sample_metrics(x, x)
        np.testing.assert_allclose(mse, 0, atol=1e-12)
        np.testing.assert_allclose(rho, 1, atol=1e-6)

    def test_per_class_and_overall(self):
        torch.manual_seed(0)
        N = 12
        x0 = torch.rand(N, 2, 16, 16) * 0.5 + 0.25
        norms = torch.tensor([[-1.0, 1.0]] * N)
        ng = torch.ones(N, 16, 16)
        ds = TensorDataset(x0, norms, ng)
        targets = np.array([0] * 6 + [1] * 6)
        net = nn.Sequential(nn.Identity())  # identity decoder: prediction = degraded input
        deg = Degradation(sparsity=0.7, noise_std=0.0, device="cpu")
        res = score_split(net, deg, ds, np.arange(N), targets, device="cpu", batch_size=4)
        self.assertEqual(res["overall"]["n"], N)
        self.assertEqual(res["per_class"][0]["n"], 6)
        self.assertEqual(res["per_class"][1]["n"], 6)
        self.assertAlmostEqual(res["overall"]["mse"], np.mean([res["per_class"][0]["mse"], res["per_class"][1]["mse"]]), places=6)


if __name__ == "__main__":
    unittest.main()
