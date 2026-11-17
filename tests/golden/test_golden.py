import os
import sys
import json
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))
import golden_common as gc  # noqa: E402

HAVE_DATA = os.path.isdir(gc.DATA) and os.path.isfile(gc.CKPT)


@unittest.skipUnless(HAVE_DATA, "data-full or Model 3 checkpoint not present")
class TestGolden(unittest.TestCase):
    def test_matches_golden(self):
        with open(gc.GOLDEN) as f:
            golden = json.load(f)
        self.assertEqual(gc.compare(gc.compute(), golden), [])

    def test_perturbed_weight_is_detected(self):
        with open(gc.GOLDEN) as f:
            golden = json.load(f)
        dec = gc.Decoder()
        dec.load_state_dict(torch.load(gc.CKPT, map_location="cpu", weights_only=True))
        with torch.no_grad():
            dec.final_conv[2].weight[:, :8] += 0.05  # last 1x1 conv, 8 input channels
        bad = gc.compare(gc.compute(decoder=dec), golden)
        self.assertTrue(bad, "a perturbed weight must break the golden check")


if __name__ == "__main__":
    unittest.main()
