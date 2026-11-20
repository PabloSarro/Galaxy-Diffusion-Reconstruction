"""Non-learned baselines on a split, at each sigma (task 3.1).

- raw:  the degraded input itself (no reconstruction).
- knn:  nearest-neighbour interpolation of the known pixels (same method as classification.py,
        moved here without logic change; scipy griddata 'nearest' per image and channel).
Both use the same degradation draw per sigma (torch seed 0), so they are comparable to the decoders.
"""
import numpy as np
import torch
from scipy.interpolate import griddata
from torch.utils.data import DataLoader, Subset

from eval_split import per_sample_metrics


def knn_reconstruct_batch(x_noised, masks):
    """x_noised: (B, C, H, W) numpy; masks: (B, H, W) bool, True = known pixel."""
    B, C, H, W = x_noised.shape
    out = np.zeros_like(x_noised)
    gx, gy = np.mgrid[0:H, 0:W]
    for i in range(B):
        m = masks[i]
        if not m.any() or m.all():
            out[i] = x_noised[i]
            continue
        pts = np.argwhere(m)
        for c in range(C):
            out[i, c] = griddata(pts, x_noised[i, c, m], (gx, gy), method="nearest")
    return out


def score_baselines(dataset, indices, targets, degradation_factory, sigmas, batch_size=32):
    """Return {sigma: {'raw': {...}, 'knn': {...}}} with overall and per-class MSE / rho."""
    res = {}
    t = np.asarray(targets)[indices]
    for sigma in sigmas:
        torch.manual_seed(0)
        deg = degradation_factory(sigma)
        loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
        acc = {k: ([], []) for k in ("raw", "knn", "raw_clip", "knn_clip")}
        for x0, norms, num_gals in loader:
            xT, masks = deg.degrade(x0, norms, num_gals)
            xk = torch.from_numpy(knn_reconstruct_batch(xT.numpy(), masks.numpy()))
            # The decoder ends in a sigmoid, so its outputs lie in [0, 1]; clipping the baselines
            # to the same range is the like-for-like comparison. Unclipped values are kept too.
            outs = {"raw": xT, "knn": xk, "raw_clip": xT.clamp(0, 1), "knn_clip": xk.clamp(0, 1)}
            for k, pred in outs.items():
                m, r = per_sample_metrics(x0, pred)
                acc[k][0].append(m); acc[k][1].append(r)
        res[float(sigma)] = {k: _summarise(np.concatenate(v[0]), np.concatenate(v[1]), t)
                             for k, v in acc.items()}
    return res


def _summarise(mse, rho, t):
    out = {"overall": {"mse": float(mse.mean()), "rho": float(rho.mean()), "n": int(len(mse))},
           "per_class": {}}
    for c in np.unique(t):
        sel = t == c
        out["per_class"][int(c)] = {"mse": float(mse[sel].mean()), "rho": float(rho[sel].mean()),
                                    "n": int(sel.sum())}
    return out
