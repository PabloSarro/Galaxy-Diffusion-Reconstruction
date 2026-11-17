"""Golden reference for the reconstruction pipeline (task 1.2).

Fixed inputs: the first 8 validation samples (split seed 42, data-full), degradation
sigma=0.22 / sparsity 0.70 with torch seed 0, Model 3 decoder checkpoint, CPU.
The output is summarised into a few numbers (not the full images) so the golden file
stays small and the check tolerates tiny float differences across machines.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dataset import GalaxyDataset  # noqa: E402
from degradation import Degradation  # noqa: E402
from helpers import generate_train_valid_datasets, set_seed  # noqa: E402
from model import Decoder  # noqa: E402

DATA = os.path.join(ROOT, "data-full")
CKPT = os.path.join(ROOT, "results", "Reconstruction", "Model 3", "MSE", "training", "decoder_best.pt")
GOLDEN = os.path.join(os.path.dirname(__file__), "golden.json")
N_SAMPLES = 8


def load_inputs():
    set_seed(42)
    dataset = GalaxyDataset(data_folder=DATA)
    _, valid = generate_train_valid_datasets(dataset, frac=0.8)
    idx = list(valid.indices[:N_SAMPLES])
    items = [valid[i] for i in range(N_SAMPLES)]
    x0 = torch.stack([it[0] for it in items])
    norms = torch.stack([it[1] for it in items])
    num_gals = torch.stack([it[2] for it in items])
    return idx, len(dataset), x0, norms, num_gals


def compute(decoder=None):
    """Return the golden summary. `decoder` lets a test pass a perturbed model."""
    idx, n_total, x0, norms, num_gals = load_inputs()
    if decoder is None:
        decoder = Decoder()
        decoder.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    decoder.eval()
    torch.manual_seed(0)
    degradation = Degradation(sparsity=0.70, noise_std=0.22, device="cpu")
    with torch.no_grad():
        xT, _ = degradation.degrade(x0, norms, num_gals)
        pred = decoder(xT)
    p = pred.double().numpy()
    return {
        "n_dataset": int(n_total),
        "valid_indices_first8": [int(i) for i in idx],
        "xT_mean": float(xT.double().mean()),
        "pred_mean": float(p.mean()),
        "pred_std": float(p.std()),
        "pred_mse_vs_x0": float(((pred - x0) ** 2).double().mean()),
        "pred_head": [float(v) for v in p.reshape(-1)[:10]],
    }


def compare(current, golden, rtol=1e-4, atol=1e-6):
    """Return a list of mismatching keys (empty list = match)."""
    bad = []
    for k, v in golden.items():
        c = current[k]
        if isinstance(v, list):
            if k == "valid_indices_first8" or k == "n_dataset":
                ok = c == v
            else:
                ok = np.allclose(c, v, rtol=rtol, atol=atol)
        else:
            ok = c == v if isinstance(v, int) else np.isclose(c, v, rtol=rtol, atol=atol)
        if not ok:
            bad.append(k)
    return bad


if __name__ == "__main__":
    # Regenerate the golden file (only when a deliberate change is approved).
    out = compute()
    with open(GOLDEN, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
