"""Task 3.2 success check: classifier accuracy on B1 reconstructions (valid split, both seeds).

Uses the committed classify_split functions. Matched sigma: 0.22 and 0. Test split is not loaded.
"""
import json, os
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from dataset import GalaxyDataset
from degradation import Degradation
from model import Decoder
from splits import load_split
from classify_split import predict, metrics, chi_transform

RUNS = {"sigma0.22": (0.22, "../results/B1/sigma0.22"), "sigma0": (0.0, "../results/B1/sigma0")}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = GalaxyDataset("../data-full/")
split, _ = load_split("../results/splits/split_v1.json")
idx = split["valid"]
arch = torch.load("../results/network_v8_1.pth", map_location=dev, weights_only=False)
out = {}
for cfg, (sig, folder) in RUNS.items():
    for seed in (1, 2):
        dec = Decoder().to(dev).eval()
        dec.load_state_dict(torch.load(f"{folder}/seed{seed}/MSE/training/decoder_best.pt", map_location=dev, weights_only=True))
        torch.manual_seed(0)
        deg = Degradation(0.70, sig, dev)
        outs = []
        with torch.no_grad():
            for x, nrm, ng in DataLoader(Subset(d, idx), batch_size=32, shuffle=False):
                x, nrm, ng = x.to(dev), nrm.to(dev), ng.to(dev)
                xT, _ = deg.degrade(x, nrm, ng)
                pred = dec(xT).cpu().numpy()
                lo, hi = nrm[:, 0, None, None, None].cpu().numpy(), nrm[:, 1, None, None, None].cpu().numpy()
                outs.append(pred * (hi - lo) + lo)
        g = np.concatenate(outs)
        t, p = predict(arch, chi_transform(g), d.targets[idx])
        out[f"{cfg}_seed{seed}"] = metrics(t, p)
        print(cfg, seed, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[f"{cfg}_seed{seed}"].items()}, flush=True)
os.makedirs("../results/B1/closing", exist_ok=True)
json.dump({"results": out, "split_file": "../results/splits/split_v1.json", "torch": torch.__version__},
          open("../results/B1/closing/classifier_b1_valid.json", "w"), indent=2)
print("saved")
