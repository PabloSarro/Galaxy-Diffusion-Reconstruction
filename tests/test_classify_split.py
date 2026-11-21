import os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from classify_split import predict, metrics, chi_transform  # noqa: E402


class FakeArch:
    """Stands in for the netloader classifier: predicts the true class exactly."""
    transforms = {"inputs": lambda x: x, "targets": lambda y: y}

    def predict(self, loader):
        preds, targets = [], []
        for _, y, _ in loader:
            y = np.asarray(y).astype(int)
            preds.append(np.eye(7)[y])
            targets.append(y)
        return {"preds": np.concatenate(preds), "targets": np.concatenate(targets)}


class TestClassifySplit(unittest.TestCase):
    def test_chi_transform_is_bounded(self):
        g = np.random.default_rng(0).normal(0, 0.5, (4, 2, 8, 8))
        chi = chi_transform(g)
        self.assertTrue(np.all(np.sqrt((chi ** 2).sum(axis=1)) <= 1 + 1e-9))

    def test_perfect_classifier_scores_one(self):
        y = np.array([0, 1, 2, 3, 4, 5, 6] * 4)
        true, pred = predict(FakeArch(), np.zeros((len(y), 2, 4, 4)), y)
        m = metrics(true, pred)
        self.assertAlmostEqual(m["acc"], 1.0)
        self.assertAlmostEqual(m["f1_macro"], 1.0)

    def test_constant_predictor_chance_is_majority_share(self):
        true = np.array([0] * 6 + [1] * 2)
        m = metrics(true, np.zeros_like(true))
        self.assertAlmostEqual(m["chance_acc"], 0.75)


if __name__ == "__main__":
    unittest.main()
