"""Driver: kNN and raw-degraded baselines per sigma on a saved split (task 3.1)."""
import argparse
import json
import os

import torch

from baselines import score_baselines
from dataset import GalaxyDataset
from degradation import Degradation
from splits import load_split

SIGMAS = [0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.22, 0.3]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split-file", required=True)
    p.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    p.add_argument("--data-folder", default="../data-full/")
    p.add_argument("--sparsity", type=float, default=0.70)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    dataset = GalaxyDataset(data_folder=a.data_folder)
    split, meta = load_split(a.split_file)
    res = score_baselines(
        dataset, split[a.split], dataset.targets,
        degradation_factory=lambda s: Degradation(sparsity=a.sparsity, noise_std=s, device="cpu"),
        sigmas=SIGMAS,
    )
    res["meta"] = {"split": a.split, "split_file": a.split_file, "sparsity": a.sparsity,
                   "torch": torch.__version__, "sigmas": SIGMAS}
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, "w") as f:
        json.dump(res, f, indent=2)
    for s in SIGMAS:
        r = res[s]
        print(f"sigma {s}: raw rho {r['raw']['overall']['rho']:.4f} | knn rho {r['knn']['overall']['rho']:.4f} "
              f"mse {r['knn']['overall']['mse']:.2e}")
