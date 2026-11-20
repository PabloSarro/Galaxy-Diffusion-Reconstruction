"""Diagnostic (task 3.1): why kNN has higher MSE than raw at sigma=0.005 on the valid split."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from baselines import knn_reconstruct_batch
from dataset import GalaxyDataset
from degradation import Degradation
from splits import load_split

d = GalaxyDataset("../data-full/")
sp, _ = load_split("../results/splits/split_v1.json")
idx = sp["valid"][:300]
torch.manual_seed(0)
deg = Degradation(0.7, 0.005, "cpu")
rows = []
for x0, n, g in DataLoader(Subset(d, idx), batch_size=32):
    xT, m = deg.degrade(x0, n, g)
    xk = torch.from_numpy(knn_reconstruct_batch(xT.numpy(), m.numpy()))
    known = m.unsqueeze(1).expand_as(x0)
    for i in range(x0.shape[0]):
        k = known[i]; r = ~k
        e_raw = (x0[i] - xT[i]) ** 2
        e_knn = (x0[i] - xk[i]) ** 2
        rows.append([e_raw[k].mean().item(), e_raw[r].mean().item(),
                     e_knn[k].mean().item(), e_knn[r].mean().item(),
                     e_raw.mean().item(), e_knn.mean().item(),
                     (n[i][1] - n[i][0]).item(), (x0[i].max() - x0[i].min()).item()])
R = np.array(rows)
cols = ["raw_known", "raw_removed", "knn_known", "knn_removed", "raw_all", "knn_all", "b-a", "x0_range"]
print("MEDIAN", dict(zip(cols, np.median(R, 0).round(5))))
print("MEAN  ", dict(zip(cols, R.mean(0).round(5))))
print("share of samples with knn_all > raw_all:", float((R[:, 5] > R[:, 4]).mean()).__round__(3))
o = np.argsort(-R[:, 5])[:5]
print("worst 5 by knn_all:")
for i in o:
    print("  knn_all %.4f raw_all %.4f b-a %.4f | knn_removed %.4f raw_removed %.4f" % (R[i, 5], R[i, 4], R[i, 6], R[i, 3], R[i, 1]))
