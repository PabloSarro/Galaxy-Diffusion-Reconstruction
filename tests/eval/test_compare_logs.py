import os, sys, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from compare_logs import parse_classification, parse_pixel

CLS = """------ Predicting: 1. Noise-Free ------
  -> Metrics: Acc: 0.612 | Prec: 0.586 | Rec: 0.588 | F1: 0.580
------ Predicting: 4. kNN ------
  -> Metrics: Acc: 0.206 | Prec: 0.1 | Rec: 0.2 | F1: 0.168
"""
PIX = """2. Direct Prediction: xT -> hat[x0]
MSE     : 0.000023
Pearson : 0.9489
"""


class TestCompare(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(parse_classification(CLS), {"1. Noise-Free": (0.612, 0.580), "4. kNN": (0.206, 0.168)})

    def test_pixel(self):
        self.assertEqual(parse_pixel(PIX), (2.3e-05, 0.9489))
        self.assertIsNone(parse_pixel("nothing"))


if __name__ == "__main__":
    unittest.main()
