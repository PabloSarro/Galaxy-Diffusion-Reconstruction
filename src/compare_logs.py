"""Compare re-run evaluation logs with the original ones (task 1.1).

Usage: python compare_logs.py NEW_CLASSIFICATION_LOG OLD_CLASSIFICATION_LOG
       python compare_logs.py --pixel NEW_PIXEL_LOG OLD_TRAIN_LOG
"""
import re
import sys


def parse_classification(text):
    """{set name: (acc, f1)} from a classification.py log."""
    out, name = {}, None
    for line in text.splitlines():
        m = re.match(r"-+ Predicting: (.+?) -+", line)
        if m:
            name = m.group(1)
        m = re.search(r"Metrics: Acc: ([\d.]+) \| Prec: [\d.]+ \| Rec: [\d.]+ \| F1: ([\d.]+)", line)
        if m and name:
            out[name] = (float(m.group(1)), float(m.group(2)))
    return out


def parse_pixel(text):
    """(mse, pearson) of the 'Direct Prediction' block of a pixel-eval / train log."""
    m = re.search(r"Direct Prediction.*?MSE\s*:\s*([\d.eE+-]+)\s*Pearson\s*:\s*([\d.]+)", text, re.S)
    return (float(m.group(1)), float(m.group(2))) if m else None


def main(argv):
    if argv[0] == "--pixel":
        new, old = (parse_pixel(open(p).read()) for p in argv[1:3])
        print(f"new MSE {new[0]:.2e} rho {new[1]:.4f} | old MSE {old[0]:.2e} rho {old[1]:.4f}")
        return
    new, old = (parse_classification(open(p).read()) for p in argv[:2])
    print(f"{'set':22s} {'acc new/old':>14s} {'F1 new/old':>14s}")
    for k, (a, f) in new.items():
        oa, of = old.get(k, (float('nan'),) * 2)
        print(f"{k:22s} {a:6.3f}/{oa:6.3f} {f:7.3f}/{of:6.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
