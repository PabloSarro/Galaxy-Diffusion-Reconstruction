"""Per-sample and per-class pixel metrics on a saved split (task 3.1).

Full split, no batch limit. Uses the grouped split file (results/splits/split_v1.json).
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def per_sample_metrics(x0, x_pred):
    """MSE and Pearson rho per sample. Inputs (N, C, H, W) tensors."""
    x0 = x0.flatten(1).double()
    x_pred = x_pred.flatten(1).double()
    mse = ((x0 - x_pred) ** 2).mean(dim=1)
    xc = x0 - x0.mean(dim=1, keepdim=True)
    pc = x_pred - x_pred.mean(dim=1, keepdim=True)
    rho = (xc * pc).sum(dim=1) / (torch.sqrt((xc ** 2).sum(dim=1) * (pc ** 2).sum(dim=1)) + 1e-12)
    return mse.numpy(), rho.numpy()


@torch.no_grad()
def score_split(decoder, degradation, dataset, indices, targets, device="cpu", batch_size=32):
    """Return {'overall': {...}, 'per_class': {cls: {...}}} for the samples in `indices`."""
    decoder.eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    mses, rhos = [], []
    torch.manual_seed(0)  # same degradation noise for every decoder that is scored
    for x0, norms, num_gals in loader:
        x0, norms, num_gals = x0.to(device), norms.to(device), num_gals.to(device)
        xT, _ = degradation.degrade(x0, norms, num_gals)
        pred = decoder(xT)
        m, r = per_sample_metrics(x0.cpu(), pred.cpu())
        mses.append(m)
        rhos.append(r)
    mse = np.concatenate(mses)
    rho = np.concatenate(rhos)
    t = np.asarray(targets)[indices]
    out = {"overall": {"mse": float(mse.mean()), "rho": float(rho.mean()), "n": int(len(mse))},
           "per_class": {}}
    for c in np.unique(t):
        sel = t == c
        out["per_class"][int(c)] = {"mse": float(mse[sel].mean()), "rho": float(rho[sel].mean()),
                                    "n": int(sel.sum())}
    return out
