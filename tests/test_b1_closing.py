import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from b1_closing import aggregate  # noqa: E402


class TestAggregate(unittest.TestCase):
    def test_mean_and_spread(self):
        per_seed = {1: {"overall": {"mse": 1.0, "rho": 0.8}}, 2: {"overall": {"mse": 3.0, "rho": 0.9}}}
        t = aggregate(per_seed)
        self.assertAlmostEqual(t["mse_mean"], 2.0)
        self.assertAlmostEqual(t["mse_spread"], 2.0)
        self.assertAlmostEqual(t["rho_mean"], 0.85)
        self.assertAlmostEqual(t["rho_spread"], 0.1)
        self.assertEqual(t["n_seeds"], 2)


if __name__ == "__main__":
    unittest.main()
