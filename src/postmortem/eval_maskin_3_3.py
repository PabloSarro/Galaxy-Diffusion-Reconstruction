"""Task 3.3 mask-input ablation: evaluate baseline (2-channel, random t) and mask-input (3-channel)
checkpoints on the same 64 VALID maps, under the realistic sampler (fresh mask and noise at each
Algorithm 2 step) and the direct prediction at T. Also the matched-randomness Algorithm 2 for
reference (oracle). Output JSON per checkpoint."""
import argparse, json, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cold_causes_3_3 as cc
from time_comparison_3_3 import make_decoder, observed_mask
from dataset import GalaxyDataset
from splits import load_split
from eval_split import per_sample_metrics


class WithMask:
    """Decoder wrapper for run_algo2: the observed mask is fixed (it comes from the observation)."""
    def __init__(self, dec, m):
        self.dec, self.m = dec, m
    def __call__(self, x, t):
        return self.dec(torch.cat([x, self.m], dim=1), t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--maskin", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = GalaxyDataset("../../data-full/")
    split, _ = load_split("../../results/splits/split_v1.json")
    idx = split["valid"][:: max(1, len(split["valid"]) // 64)][:64]
    dec = make_decoder(a.maskin).to(dev).eval()
    dec.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=True))
    diff = cc.Diffusion(timesteps=cc.T, max_sparsity=cc.SP, max_noise_std=cc.STD, device=dev)
    acc = {"direct": ([], []), "alg2_resample": ([], []), "alg2_matched_oracle": ([], [])}
    for bi, (x0, norms, ng) in enumerate(DataLoader(Subset(ds, idx), batch_size=32, shuffle=False)):
        x0, norms, ng = x0.to(dev), norms.to(dev), ng.to(dev)
        B, _, H, W = x0.shape
        g = torch.Generator(device=dev).manual_seed(100 + bi)
        M = torch.rand((B, H, W), device=dev, generator=g)
        E1 = torch.randn((B, H, W), device=dev, generator=g)
        E2 = torch.randn((B, H, W), device=dev, generator=g)
        tT = torch.full((B,), cc.T, device=dev, dtype=torch.long)
        with torch.no_grad():
            xT = diff.degrade(x0, norms, ng, tT, fixed_mask_rand=M, fixed_e1=E1, fixed_e2=E2)
            obs = observed_mask(tT, ng, dev, cc.SP, M)
            net = WithMask(dec, obs) if a.maskin else dec
            inp = torch.cat([xT, obs], 1) if a.maskin else xT
            preds = {
                "direct": dec(inp, tT),
                "alg2_resample": cc.run_algo2(net, diff, xT, norms, ng, 1000, "resample"),
                "alg2_matched_oracle": cc.run_algo2(net, diff, xT, norms, ng, 1000, "matched", M, E1, E2),
            }
        for k, p in preds.items():
            m, r = per_sample_metrics(x0.cpu(), p.cpu())
            acc[k][0].append(m); acc[k][1].append(r)
        print("batch", bi, flush=True)
    res = {"ckpt": a.ckpt, "maskin": a.maskin, "n": len(idx),
           **{k: {"mse": float(np.concatenate(v[0]).mean()), "rho": float(np.concatenate(v[1]).mean())} for k, v in acc.items()}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
