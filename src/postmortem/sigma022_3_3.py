"""Task 3.3 follow-up: cold sampler at sigma = 0.22 with partial knowledge of the randomness.

Degradation (x_T and the sampler) uses max_noise_std = 0.22. The sampler's per-step randomness:
  noise_known  : ellipticity noise (E1, E2) is the TRUE realisation, the mask is resampled each step
  mask_known   : mask is the TRUE one, noise resampled each step (for contrast)
  realistic    : both resampled each step
  oracle       : both the TRUE mask and noise at every step
  direct       : one decoder call at T
Checkpoint: archived ep50_sp0.2_std0.005 (trained at sigma 0.005; sigma 0.22 is out of its training range).
"""
import json, os
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
import cold_causes_3_3 as cc
from dataset import GalaxyDataset
from splits import load_split
from eval_split import per_sample_metrics

SIG = 0.22


def sample(dec, diff, xT, norms, ng, K, fix_mask, fix_noise, M, E1, E2):
    B, _, H, W = xT.shape
    dev = xT.device
    x = xT.clone()
    ts = [round(k * cc.T / K) for k in range(K, -1, -1)]
    with torch.no_grad():
        for k in range(K):
            th = torch.full((B,), ts[k], dtype=torch.long, device=dev)
            tl = torch.full((B,), ts[k + 1], dtype=torch.long, device=dev)
            x0p = dec(x, th)
            mm = M if fix_mask else torch.rand((B, H, W), device=dev)
            if fix_noise:
                e1, e2 = E1, E2
            else:
                e1, e2 = torch.randn((B, H, W), device=dev), torch.randn((B, H, W), device=dev)
            xh = diff.degrade(x0p, norms, ng, th, fixed_mask_rand=mm, fixed_e1=e1, fixed_e2=e2)
            xl = diff.degrade(x0p, norms, ng, tl, fixed_mask_rand=mm, fixed_e1=e1, fixed_e2=e2)
            x = x - xh + xl
    return x


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = GalaxyDataset("../../data-full/")
    split, _ = load_split("../../results/splits/split_v1.json")
    idx = split["valid"][:: max(1, len(split["valid"]) // 64)][:64]
    dec = cc.OldDecoder(timesteps=cc.T).to(dev).eval()
    dec.load_state_dict(torch.load(cc.CKPT, map_location=dev, weights_only=True))
    diff = cc.Diffusion(timesteps=cc.T, max_sparsity=cc.SP, max_noise_std=SIG, device=dev)
    conds = {"direct": None, "noise_known": (False, True), "mask_known": (True, False),
             "realistic": (False, False), "oracle": (True, True)}
    acc = {k: ([], []) for k in conds}
    for bi, (x0, norms, ng) in enumerate(DataLoader(Subset(ds, idx), batch_size=32, shuffle=False)):
        x0, norms, ng = x0.to(dev), norms.to(dev), ng.to(dev)
        B, _, H, W = x0.shape
        g = torch.Generator(device=dev).manual_seed(200 + bi)
        M = torch.rand((B, H, W), device=dev, generator=g)
        E1 = torch.randn((B, H, W), device=dev, generator=g)
        E2 = torch.randn((B, H, W), device=dev, generator=g)
        with torch.no_grad():
            xT = diff.degrade(x0, norms, ng, torch.full((B,), cc.T, device=dev, dtype=torch.long),
                              fixed_mask_rand=M, fixed_e1=E1, fixed_e2=E2)
            for name, cfg in conds.items():
                if cfg is None:
                    p = dec(xT, torch.full((B,), cc.T, device=dev, dtype=torch.long))
                else:
                    p = sample(dec, diff, xT, norms, ng, 1000, cfg[0], cfg[1], M, E1, E2)
                m, r = per_sample_metrics(x0.cpu(), p.cpu())
                acc[name][0].append(m); acc[name][1].append(r)
        print("batch", bi, flush=True)
    res = {"meta": {"sigma": SIG, "ckpt": "ep50_sp0.2_std0.005 (trained sigma 0.005)", "n": len(idx), "sparsity": cc.SP},
           **{k: {"mse": float(np.concatenate(v[0]).mean()), "rho": float(np.concatenate(v[1]).mean())} for k, v in acc.items()}}
    os.makedirs("../../results/postmortem_3_3", exist_ok=True)
    json.dump(res, open("../../results/postmortem_3_3/sigma022_valid.json", "w"), indent=2)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
