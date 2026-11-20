"""Physical-space metrics at matched sigma on the valid split (task 3.1).

For raw input, kNN and each decoder: MSE and rho in physical shear, chi space, and
Kaiser-Squires convergence (kappa) space, plus the relative error of the kappa power spectrum
(radially averaged, 12 bins). Medians are reported next to means (see TASKS.md).
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from baselines import knn_reconstruct_batch
from dataset import GalaxyDataset
from degradation import Degradation
from helpers import shear_to_mass
from model import Decoder
from splits import load_split

NBINS = 12


def chi(g):
    return 2 * g / (1 + (g ** 2).sum(dim=1, keepdim=True))


def rho_per_sample(a, b):
    a = a.flatten(1).double(); b = b.flatten(1).double()
    ac = a - a.mean(1, keepdim=True); bc = b - b.mean(1, keepdim=True)
    return ((ac * bc).sum(1) / (torch.sqrt((ac ** 2).sum(1) * (bc ** 2).sum(1)) + 1e-12)).numpy()


def mse_per_sample(a, b):
    return ((a - b) ** 2).flatten(1).double().mean(1).numpy()


def radial_power(kappa):
    """kappa (N,1,H,W) -> (N, NBINS) radially averaged power."""
    k = torch.fft.fftshift(torch.fft.fft2(kappa[:, 0]), dim=(-2, -1))
    P = (k.abs() ** 2).numpy()
    N, H, W = P.shape
    yy, xx = np.meshgrid(np.arange(H) - H // 2, np.arange(W) - W // 2, indexing="ij")
    r = np.sqrt(xx ** 2 + yy ** 2)
    edges = np.linspace(1, r.max(), NBINS + 1)
    out = np.zeros((N, NBINS))
    for j in range(NBINS):
        sel = (r >= edges[j]) & (r < edges[j + 1])
        out[:, j] = P[:, sel].mean(axis=1)
    return out


def ps_rel_err(kappa_pred, kappa_true):
    Pp = radial_power(kappa_pred); Pt = radial_power(kappa_true)
    return np.mean(np.abs(Pp - Pt) / (Pt + 1e-30), axis=1)


def score(name, g_pred, g_true, store):
    chi_p, chi_t = chi(g_pred), chi(g_true)
    k_p = shear_to_mass(g_pred); k_t = shear_to_mass(g_true)
    store[name] = {
        "phys_mse": mse_per_sample(g_pred, g_true), "phys_rho": rho_per_sample(g_pred, g_true),
        "chi_mse": mse_per_sample(chi_p, chi_t), "chi_rho": rho_per_sample(chi_p, chi_t),
        "kappa_mse": mse_per_sample(k_p, k_t), "kappa_rho": rho_per_sample(k_p, k_t),
        "ps_rel_err": ps_rel_err(k_p, k_t),
    }


def summarise(store):
    out = {}
    for name, d in store.items():
        out[name] = {}
        for k, v in d.items():
            out[name][k] = {"mean": float(np.mean(v)), "median": float(np.median(v))}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-file", default="../results/splits/split_v1.json")
    p.add_argument("--sparsity", type=float, default=0.70)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    ds = GalaxyDataset(data_folder="../data-full/")
    split, _ = load_split(a.split_file)
    valid = split["valid"]
    cases = {"M1": ("1", 0.005), "M2": ("2", 0.0), "M3": ("3", 0.22)}
    store = {}
    dec = {}
    for tag, (num, _) in cases.items():
        m = Decoder()
        m.load_state_dict(torch.load(os.path.join("..", "results", "Reconstruction", f"Model {num}", "MSE", "training", "decoder_best.pt"), map_location="cpu", weights_only=True))
        dec[tag] = m.eval()
    for tag, (num, sigma) in cases.items():
        torch.manual_seed(0)
        deg = Degradation(sparsity=a.sparsity, noise_std=sigma, device="cpu")
        G0, GR, GK, GU = [], [], [], []
        for x0, norms, ng in DataLoader(Subset(ds, valid), batch_size=64, shuffle=False):
            xT, masks = deg.degrade(x0, norms, ng)
            aa = norms[:, 0].view(-1, 1, 1, 1); bb = norms[:, 1].view(-1, 1, 1, 1)
            with torch.no_grad():
                xu = dec[tag](xT)
                xk = torch.from_numpy(knn_reconstruct_batch(xT.numpy(), masks.numpy()))
            phys = lambda x: x * (bb - aa) + aa
            G0.append(phys(x0)); GR.append(phys(xT)); GK.append(phys(xk)); GU.append(phys(xu))
        G0 = torch.cat(G0); GR = torch.cat(GR); GK = torch.cat(GK); GU = torch.cat(GU)
        score(f"{tag} raw (s={sigma})", GR, G0, store)
        score(f"{tag} knn (s={sigma})", GK, G0, store)
        score(f"{tag} unet (s={sigma})", GU, G0, store)
        print(tag, "done", flush=True)
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, "w") as f:
        json.dump({"summary": summarise(store)}, f, indent=2)
    print("saved", a.output)


if __name__ == "__main__":
    main()
