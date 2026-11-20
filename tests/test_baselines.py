import os, sys, unittest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from baselines import knn_reconstruct_batch  # noqa: E402


class TestKnn(unittest.TestCase):
    def test_known_pixels_are_kept(self):
        rng = np.random.default_rng(0)
        x = rng.random((2, 2, 12, 12))
        mask = rng.random((2, 12, 12)) < 0.3
        out = knn_reconstruct_batch(x, mask)
        for i in range(2):
            for c in range(2):
                np.testing.assert_allclose(out[i, c][mask[i]], x[i, c][mask[i]], atol=1e-12)

    def test_constant_image_stays_constant(self):
        x = np.full((1, 2, 10, 10), 0.4)
        mask = np.zeros((1, 10, 10), bool)
        mask[0, ::3, ::3] = True
        out = knn_reconstruct_batch(x, mask)
        np.testing.assert_allclose(out, 0.4)

    def test_empty_or_full_mask_is_passthrough(self):
        x = np.random.default_rng(1).random((2, 2, 8, 8))
        mask = np.stack([np.zeros((8, 8), bool), np.ones((8, 8), bool)])
        np.testing.assert_array_equal(knn_reconstruct_batch(x, mask), x)


class TestClipping(unittest.TestCase):
    def test_clipping_never_increases_error_for_targets_in_unit_range(self):
        from eval_split import per_sample_metrics
        rng = np.random.default_rng(2)
        x0 = torch.from_numpy(rng.random((4, 2, 8, 8)).astype(np.float32))
        pred = x0 + torch.from_numpy(rng.normal(0, 2.0, (4, 2, 8, 8)).astype(np.float32))
        m_raw, _ = per_sample_metrics(x0, pred)
        m_clip, _ = per_sample_metrics(x0, pred.clamp(0, 1))
        self.assertTrue(np.all(m_clip <= m_raw + 1e-12))


if __name__ == "__main__":
    unittest.main()
