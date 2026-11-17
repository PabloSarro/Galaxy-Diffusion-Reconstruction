"""Resume test (task 2.2): an interrupted run resumed from last.pt must equal an uninterrupted run."""
import os
import shutil
import sys
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from degradation import Degradation  # noqa: E402
from training import run_training, LAST_NAME, BEST_NAME  # noqa: E402


def _tiny_setup(seed):
    """Small synthetic data and a tiny 2-channel model, so training takes seconds."""
    g = torch.Generator().manual_seed(123)
    N, H, W = 24, 16, 16
    x0 = torch.rand((N, 2, H, W), generator=g) * 0.5 + 0.25
    norms = torch.tensor([[-1.0, 1.0]] * N)
    num_gals = torch.ones((N, H, W))
    ds = TensorDataset(x0, norms, num_gals)
    train, valid = torch.utils.data.random_split(
        ds, [18, 6], generator=torch.Generator().manual_seed(0))
    loader_gen = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train, batch_size=4, shuffle=True, generator=loader_gen)
    valid_loader = DataLoader(valid, batch_size=4, shuffle=False)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Conv2d(2, 8, 3, padding=1), nn.SiLU(), nn.Conv2d(8, 2, 3, padding=1), nn.Sigmoid())
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    deg = Degradation(sparsity=0.7, noise_std=0.1, device="cpu")
    return model, opt, deg, train_loader, valid_loader, loader_gen


def _run(out_dir, epochs, seed, resume):
    model, opt, deg, tl, vl, gen = _tiny_setup(seed)
    losses = run_training(model, opt, deg, tl, vl, gen, epochs=epochs, out_dir=out_dir,
                          loss_method="MSE", device="cpu", resume=resume, log=lambda *_: None)
    return model, losses


class TestResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_interrupted_then_resumed_equals_uninterrupted(self):
        ref_dir = os.path.join(self.tmp, "ref")
        ref_model, ref_losses = _run(ref_dir, epochs=3, seed=7, resume=False)

        run_dir = os.path.join(self.tmp, "run")
        _run(run_dir, epochs=1, seed=7, resume=False)          # "job killed" after epoch 1
        self.assertTrue(os.path.isfile(os.path.join(run_dir, LAST_NAME)))
        self.assertTrue(os.path.isfile(os.path.join(run_dir, BEST_NAME)))
        # Resume in a fresh process-like state: a different global seed must not matter.
        res_model, res_losses = _run(run_dir, epochs=3, seed=999, resume=True)

        for (k, a), (_, b) in zip(ref_model.state_dict().items(), res_model.state_dict().items()):
            self.assertTrue(torch.equal(a, b), f"weights differ at {k}")
        self.assertEqual(ref_losses[0], res_losses[0])
        self.assertEqual(ref_losses[1], res_losses[1])

    def test_resume_without_checkpoint_starts_fresh(self):
        fresh_dir = os.path.join(self.tmp, "fresh")
        model, losses = _run(fresh_dir, epochs=2, seed=7, resume=True)
        self.assertEqual(len(losses[0]), 2)

    def test_last_checkpoint_is_written_each_epoch(self):
        d = os.path.join(self.tmp, "each")
        _run(d, epochs=2, seed=7, resume=False)
        ckpt = torch.load(os.path.join(d, LAST_NAME), weights_only=False)
        self.assertEqual(ckpt["epoch"], 2)
        self.assertFalse(os.path.exists(os.path.join(d, LAST_NAME + ".tmp")))


if __name__ == "__main__":
    unittest.main()
