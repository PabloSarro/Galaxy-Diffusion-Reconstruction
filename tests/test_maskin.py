import os, sys, unittest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "postmortem"))
import time_comparison_3_3 as tc  # noqa: E402


class TestMaskIn(unittest.TestCase):
    def test_observed_mask_excludes_removed_and_empty(self):
        B, H, W = 2, 4, 4
        ng = torch.ones(B, H, W); ng[0, 0, :] = 0           # row 0 of image 0 has no galaxies
        t = torch.tensor([1000, 1000])                        # sparsity 0.7 at T
        mr = torch.ones(B, H, W)                              # 1.0 = never removed (removed if mr < 0.7)
        mr[1, :, 1] = 0.1                                     # column 1 of image 1 removed
        m = tc.observed_mask(t, ng, "cpu", 0.7, mr)
        self.assertEqual(m.shape, (B, 1, H, W))
        self.assertTrue(torch.all(m[0, 0, 0] == 0))           # no galaxies
        self.assertTrue(torch.all(m[0, 0, 1:] == 1))
        self.assertTrue(torch.all(m[1, 0, :, 1] == 0))        # removed
        self.assertTrue(torch.all(m[1, 0, :, 0] == 1))

    def test_maskin_decoder_takes_three_channels(self):
        torch.manual_seed(0)
        d = tc.make_decoder(maskin=True).eval()
        with torch.no_grad():
            out = d(torch.rand(1, 3, 32, 32), torch.tensor([500]))
        self.assertEqual(tuple(out.shape), (1, 2, 32, 32))


if __name__ == "__main__":
    unittest.main()
