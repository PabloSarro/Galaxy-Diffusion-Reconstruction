import argparse, os, sys, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from eval_args import add_eval_args, DEFAULT_NOISE_STDS


class TestEvalArgs(unittest.TestCase):
    def parse(self, argv):
        return add_eval_args(argparse.ArgumentParser()).parse_args(argv)

    def test_defaults_keep_old_behaviour(self):
        a = self.parse(["--decoder-path", "d.pt", "--output-dir", "o"])
        self.assertEqual(a.sparsity, 0.70)
        self.assertEqual(a.noise_stds, DEFAULT_NOISE_STDS)
        self.assertEqual((a.batch_size, a.seed, a.data_folder), (32, 42, "../data-full/"))

    def test_required_and_overrides(self):
        with self.assertRaises(SystemExit):
            self.parse([])
        a = self.parse(["--decoder-path", "d.pt", "--output-dir", "o", "--noise-stds", "0.22", "0.3"])
        self.assertEqual(a.noise_stds, [0.22, 0.3])


if __name__ == "__main__":
    unittest.main()
