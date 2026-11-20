"""Classifier scoring on a saved split (task 3.1). Same pipeline as classification.py, but:
  - scores the saved split (valid by default), not the old random validation split;
  - only the matched-sigma cases needed for the 3.1 table, plus the provenance check.

Cases (--cases):
  clean_train    200 random TRAIN-split maps, noise-free (provenance: does the classifier know them?)
  clean_valid    all VALID maps, noise-free (ceiling)
  noisy:SIG      degraded input at sigma SIG
  knn:SIG        kNN reconstruction at sigma SIG
  unet:N:SIG     decoder of Model N (results/Reconstruction/Model N) applied to the degraded input at sigma SIG
Chance is the constant-predictor accuracy on the same split.
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, Subset

from baselines import knn_reconstruct_batch
from dataset import GalaxyDataset
from degradation import Degradation
from model import Decoder
from splits import load_split

CLASSIFIER_PATH = "../results/network_v8_1.pth"


class CustomDataset(Dataset):
    def __init__(self, ids, targets, shears):
        self.ids, self.targets, self.shears = ids, targets, shears

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.targets[i], self.shears[i]


def chi_transform(red):
    return 2 * red / (1 + (red ** 2).sum(axis=1, keepdims=True))


def predict(arch, base_shears, base_targets):
    """base_shears: (N,2,H,W) physical-units numpy, chi-transformed already."""
    ds = CustomDataset(np.arange(len(base_targets)), base_targets.copy(), base_shears.copy())
    ds.shears = arch.transforms["inputs"](ds.shears)
    ds.targets = arch.transforms["targets"](ds.targets)
    out = arch.predict(DataLoader(ds, batch_size=32, shuffle=False))
    true = np.round(out["targets"]).astype(int)
    pred = np.round(np.argmax(out["preds"], axis=1)).astype(int)
    return true, pred


def metrics(true, pred):
    classes = np.unique(true)
    counts = np.array([(true == c).sum() for c in classes])
    chance = counts.max() / counts.sum()  # constant predictor = majority class
    return {"acc": float(accuracy_score(true, pred)),
            "bal_acc": float(balanced_accuracy_score(true, pred)),
            "f1_macro": float(f1_score(true, pred, average="macro", zero_division=0)),
            "chance_acc": float(chance), "n": int(len(true))}


def to_physical_chi(x, norms):
    a = norms[:, 0].numpy()[:, None, None, None]
    b = norms[:, 1].numpy()[:, None, None, None]
    return chi_transform(x.numpy() * (b - a) + a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-file", default="../results/splits/split_v1.json")
    p.add_argument("--data-folder", default="../data-full/")
    p.add_argument("--sparsity", type=float, default=0.70)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, flush=True)
    arch = torch.load(CLASSIFIER_PATH, map_location=dev, weights_only=False)
    ds = GalaxyDataset(data_folder=a.data_folder)
    split, _ = load_split(a.split_file)
    valid = split["valid"]
    rng = np.random.default_rng(0)
    train_sub = rng.choice(split["train"], size=200, replace=False)
    targets_all = ds.targets
    results = {}
    decoders = {}

    for case in a.cases:
        parts = case.split(":")
        kind = parts[0]
        if kind in ("clean_train", "clean_valid"):
            idx = train_sub if kind == "clean_train" else valid
            loader = DataLoader(Subset(ds, idx), batch_size=64, shuffle=False)
            xs, ns = [], []
            for x0, norms, _ in loader:
                xs.append(x0); ns.append(norms)
            x0 = torch.cat(xs); norms = torch.cat(ns)
            shears = to_physical_chi(x0, norms)
            true, pred = predict(arch, shears, targets_all[idx])
        else:
            sigma = float(parts[-1])
            torch.manual_seed(0)
            deg = Degradation(sparsity=a.sparsity, noise_std=sigma, device="cpu")
            xs, ns, ms, gs = [], [], [], []
            for x0, norms, ng in DataLoader(Subset(ds, valid), batch_size=64, shuffle=False):
                xT, m = deg.degrade(x0, norms, ng)
                xs.append(x0); ns.append(norms); ms.append(m); gs.append(xT)
            x0 = torch.cat(xs); norms = torch.cat(ns); masks = torch.cat(ms); xT = torch.cat(gs)
            if kind == "noisy":
                shears = to_physical_chi(xT, norms)
            elif kind == "knn":
                xk = knn_reconstruct_batch(xT.numpy(), masks.numpy())
                shears = to_physical_chi(torch.from_numpy(xk), norms)
            elif kind == "unet":
                name = parts[1]  # model number, e.g. "3"
                path = os.path.join("..", "results", "Reconstruction", f"Model {name}", "MSE", "training", "decoder_best.pt")
                if name not in decoders:
                    dec = Decoder()
                    dec.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
                    dec = dec.to(dev)
                    dec.eval()
                    decoders[name] = dec
                with torch.no_grad():
                    outs = torch.cat([decoders[name](xT[i:i + 64].to(dev)).cpu() for i in range(0, len(xT), 64)])
                shears = to_physical_chi(outs, norms)
            else:
                raise ValueError(case)
            true, pred = predict(arch, shears, targets_all[valid])
        results[case] = metrics(true, pred)
        print(case, json.dumps(results[case]), flush=True)

    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, "w") as f:
        json.dump({"results": results, "split_file": a.split_file, "torch": torch.__version__}, f, indent=2)
    print("saved", a.output)


if __name__ == "__main__":
    main()
