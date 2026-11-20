"""Diagnostic (task 3.1): cause of the 'MSE=62' baseline for Model 3 (sigma=0.22), physical units."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import GalaxyDataset
from degradation import Degradation
from splits import load_split

d = GalaxyDataset("../data-full/")
sp, _ = load_split("../results/splits/split_v1.json")
idx = sp["valid"][:512]
torch.manual_seed(0)
deg = Degradation(0.7, 0.22, "cpu")
per = []  # per sample: phys MSE, frac |g|>1 in known pixels, b-a
for x0, n, g in DataLoader(Subset(d, idx), batch_size=64):
    xT, m = deg.degrade(x0, n, g)
    a = n[:, 0].view(-1, 1, 1, 1); b = n[:, 1].view(-1, 1, 1, 1)
    gT = xT * (b - a) + a                     # observed physical shear
    g0 = x0 * (b - a) + a                     # true physical shear
    mse_phys = ((gT - g0) ** 2).flatten(1).mean(1)
    gmag = torch.sqrt(gT[:, 0] ** 2 + gT[:, 1] ** 2)
    known = m
    frac_strong = ((gmag > 1) & known).flatten(1).sum(1).float() / known.flatten(1).sum(1).clamp(min=1)
    for i in range(x0.shape[0]):
        per.append([mse_phys[i].item(), frac_strong[i].item(), (b - a)[i].item()])
P = np.array(per)
mse = P[:, 0]; fs = P[:, 1]; ba = P[:, 2]
print("phys MSE of degraded input: mean %.3f median %.4f max %.2f" % (mse.mean(), np.median(mse), mse.max()))
print("samples with frac(|g|>1) > 0: %.3f" % (fs > 0).mean())
print("corr(phys MSE, b-a)=%.3f  corr(phys MSE, frac|g|>1)=%.3f" % (np.corrcoef(mse, ba)[0, 1], np.corrcoef(mse, fs)[0, 1]))
top = np.argsort(-mse)[:10]
print("top-10 samples: phys MSE | frac|g|>1 | b-a")
for i in top:
    print("  %.2f | %.4f | %.4f" % (mse[i], fs[i], ba[i]))
print("mean MSE over samples with frac|g|>1 == 0: %.4f" % (mse[fs == 0].mean() if (fs == 0).any() else float('nan')))
