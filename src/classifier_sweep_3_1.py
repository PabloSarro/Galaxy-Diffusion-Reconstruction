"""Task 3.1 provenance/tolerance: classifier accuracy on degraded CLEAN maps vs sparsity s and sigma (no decoder)."""
import json, os
import numpy as np, torch
from dataset import GalaxyDataset
from degradation import Degradation
from splits import load_split
from classify_split import predict, metrics, chi_transform
from torch.utils.data import DataLoader, Subset

d = GalaxyDataset("../data-full/")
sp, _ = load_split("../results/splits/split_v1.json")
idx = sp["valid"]
arch = torch.load("../results/network_v8_1.pth", map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
res = {}
for s in [0.25, 0.5, 0.7]:
    for sig in [0.0, 0.005, 0.22]:
        torch.manual_seed(0)
        deg = Degradation(s, sig, "cpu")
        outs = []
        for x, nrm, ng in DataLoader(Subset(d, idx), batch_size=32, shuffle=False):
            xT, _ = deg.degrade(x, nrm, ng)
            lo, hi = nrm[:, 0, None, None, None].numpy(), nrm[:, 1, None, None, None].numpy()
            outs.append(xT.numpy() * (hi - lo) + lo)
        g = np.concatenate(outs)
        true, pred = predict(arch, chi_transform(g), d.targets[idx])
        r = metrics(true, pred)
        res[f"sparse:{s},sigma:{sig}"] = r
        print(f"s={s} sigma={sig}: acc {r['acc']:.3f} F1 {r['f1_macro']:.3f} chance {r['chance_acc']:.3f}", flush=True)
json.dump({"results": res, "split_file": "../results/splits/split_v1.json", "torch": torch.__version__},
          open("../results/split_v1/classification/classification_sparsity_sigma_valid.json", "w"), indent=2)
print("saved")
