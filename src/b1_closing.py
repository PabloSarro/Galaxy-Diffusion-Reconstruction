"""Task 3.2 closing steps: full-valid per-class scoring of the B1 checkpoints, mean/spread table,
B1 golden output, weight backup (D8). Test split is never loaded.
"""
import argparse
import json
import os
import shutil
import sys
import sys

import numpy as np
import torch

from degradation import Degradation
from dataset import GalaxyDataset
from eval_split import score_split
from model import Decoder
from splits import load_split
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests', 'golden'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests', 'golden'))
import golden_common as gc  # noqa: E402  # noqa: E402

RUNS = {  # config -> {seed: run folder}
    "sigma0.22": {1: "../results/B1/sigma0.22/seed1", 2: "../results/B1/sigma0.22/seed2"},
    "sigma0": {1: "../results/B1/sigma0/seed1", 2: "../results/B1/sigma0/seed2"},
}
SIGMA = {"sigma0.22": 0.22, "sigma0": 0.0}


def aggregate(per_seed):
    """per_seed: {seed: {'overall': {'mse','rho'}}} -> mean and spread (max-min) over seeds."""
    mse = np.array([v["overall"]["mse"] for v in per_seed.values()])
    rho = np.array([v["overall"]["rho"] for v in per_seed.values()])
    return {"mse_mean": float(mse.mean()), "mse_spread": float(mse.max() - mse.min()),
            "rho_mean": float(rho.mean()), "rho_spread": float(rho.max() - rho.min()),
            "n_seeds": len(per_seed)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-file", default="../results/splits/split_v1.json")
    p.add_argument("--out-dir", default="../results/B1/closing")
    p.add_argument("--backup-dir", default="/home/sarro/backups/B1")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = GalaxyDataset(data_folder="../data-full/")
    split, _ = load_split(a.split_file)
    valid = split["valid"]

    scores = {}
    for cfg, seeds in RUNS.items():
        scores[cfg] = {}
        for seed, folder in seeds.items():
            ck = os.path.join(folder, "MSE", "training", "decoder_best.pt")
            dec = Decoder().to(dev)
            dec.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
            deg = Degradation(sparsity=0.70, noise_std=SIGMA[cfg], device=dev)
            scores[cfg][seed] = score_split(dec, deg, ds, valid, ds.targets, device=dev, batch_size=32)
            print(cfg, seed, json.dumps(scores[cfg][seed]["overall"]), flush=True)

    table = {cfg: aggregate(scores[cfg]) for cfg in RUNS}
    with open(os.path.join(a.out_dir, "b1_valid_full.json"), "w") as f:
        json.dump({"per_seed": {c: {str(s): v for s, v in d.items()} for c, d in scores.items()},
                   "table": table}, f, indent=2)
    print(json.dumps(table, indent=2), flush=True)

    # Golden output for B1 (sigma 0.22, seed 1), same fixed inputs as tests/golden.
    dec = Decoder()
    dec.load_state_dict(torch.load(os.path.join(RUNS["sigma0.22"][1], "MSE", "training", "decoder_best.pt"),
                                   map_location="cpu", weights_only=True))
    golden = gc.compute(decoder=dec)
    with open(os.path.join(a.out_dir, "golden_B1_sigma0.22_seed1.json"), "w") as f:
        json.dump(golden, f, indent=2)
    print("golden", json.dumps({k: golden[k] for k in ("pred_mse_vs_x0", "pred_mean")}), flush=True)

    # Backup of the best checkpoints (copy, nothing deleted).
    for cfg, seeds in RUNS.items():
        for seed, folder in seeds.items():
            dst = os.path.join(a.backup_dir, cfg, f"seed{seed}")
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(os.path.join(folder, "MSE", "training", "decoder_best.pt"), dst)
    print("backup done", a.backup_dir, flush=True)


if __name__ == "__main__":
    main()
