"""Task 3.3: Algorithm 2 with K sampler steps (K in 1,5,20,100,1000) on the old checkpoint, matched noise.
Same subset, degradation and sampler as cold_causes_3_3.py (imported)."""
import json, os
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
import cold_causes_3_3 as cc
from dataset import GalaxyDataset
from splits import load_split
from eval_split import per_sample_metrics

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--noise-std", type=float, default=cc.STD)
_ap.add_argument("--out", default="../../results/postmortem_3_3/t_sweep_valid.json")
ARGS = _ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = GalaxyDataset("../../data-full/")
split, _ = load_split("../../results/splits/split_v1.json")
idx = split["valid"][:: max(1, len(split["valid"]) // 64)][:64]
dec = cc.OldDecoder(timesteps=cc.T).to(dev).eval()
dec.load_state_dict(torch.load(cc.CKPT, map_location=dev, weights_only=True))
diff = cc.Diffusion(timesteps=cc.T, max_sparsity=cc.SP, max_noise_std=ARGS.noise_std, device=dev)
Ks = [1, 5, 20, 100, 1000]
acc = {K: ([], []) for K in Ks}
for bi, (x0, norms, ng) in enumerate(DataLoader(Subset(ds, idx), batch_size=32, shuffle=False)):
    x0, norms, ng = x0.to(dev), norms.to(dev), ng.to(dev)
    B, _, H, W = x0.shape
    g = torch.Generator(device=dev).manual_seed(100 + bi)
    M = torch.rand((B, H, W), device=dev, generator=g)
    E1 = torch.randn((B, H, W), device=dev, generator=g)
    E2 = torch.randn((B, H, W), device=dev, generator=g)
    with torch.no_grad():
        xT = diff.degrade(x0, norms, ng, torch.full((B,), cc.T, device=dev, dtype=torch.long),
                          fixed_mask_rand=M, fixed_e1=E1, fixed_e2=E2)
        for K in Ks:
            p = cc.run_algo2(dec, diff, xT, norms, ng, K, "matched", M, E1, E2)
            m, r = per_sample_metrics(x0.cpu(), p.cpu())
            acc[K][0].append(m); acc[K][1].append(r)
res = {"meta": {"n": len(idx), "matched randomness": True, "torch": torch.__version__},
       "K": {str(K): {"mse": float(np.concatenate(acc[K][0]).mean()), "rho": float(np.concatenate(acc[K][1]).mean())} for K in Ks}}
print(json.dumps(res, indent=1))
os.makedirs("../../results/postmortem_3_3", exist_ok=True)
res["meta"]["noise_std"] = ARGS.noise_std
json.dump(res, open(ARGS.out, "w"), indent=2)
