"""Task 3.3: cold-diffusion post-mortem on the OLD checkpoint (ep50_sp0.2_std0.005).

Uses the archived 7b9b37c code (archive/diffusion_7b9b37c/). Tests each candidate cause with a
single change against a matched baseline:
  matched        sampler gets the same mask and noise that produced x_T (old default behaviour)
  mask_mismatch  sampler mask differs from the one that made x_T (noise matched)
  noise_mismatch sampler noise differs from the one that made x_T (mask matched)
  resample       sampler draws new mask and noise at every step (noise treated as not reversible)
  direct         one decoder call at t=T (no iteration)
T sweep: Algorithm 2 with K steps, K in {1,5,20,100,1000}, same endpoints (t/T schedule).
Data: data-full VALID split (old BAHAMAS data is gone, D12): absolute numbers are not comparable
with the old report; the comparison between conditions is.
"""
import json, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util
ARCH = os.path.join(HERE, "..", "..", "archive", "diffusion_7b9b37c")
def _load(name, path):  # load the archived 7b9b37c modules without clashing with src/
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
Diffusion = _load("old_diffusion", os.path.join(ARCH, "diffusion.py")).Diffusion
OldDecoder = _load("old_model", os.path.join(ARCH, "model.py")).Decoder  # time-conditioned
sys.path.insert(0, os.path.join(HERE, ".."))
from dataset import GalaxyDataset
from splits import load_split
from eval_split import per_sample_metrics

CKPT = os.path.join(HERE, "..", "..", "archive", "results", "old_reconstr", "2-new_arch", "ep50_sp0.2_std0.005", "training", "decoder_best.pt")
SP, STD, T = 0.20, 0.005, 1000


def run_algo2(dec, diff, xT, norms, ng, K, mode, M=None, E1=None, E2=None, gen=None):
    """Algorithm 2 with K steps. mode: matched | resample. M,E1,E2: randomness used for every step."""
    B, C, H, W = xT.shape
    dev = xT.device
    x = xT.clone()
    ts = [round(k * T / K) for k in range(K, -1, -1)]  # T, ..., 0
    with torch.no_grad():
        for k in range(K):
            t_hi, t_lo = ts[k], ts[k + 1]
            tb = torch.full((B,), t_hi, dtype=torch.long, device=dev)
            x0p = dec(x, tb)
            if mode == "resample":
                m_, e1_, e2_ = None, None, None
            else:
                m_, e1_, e2_ = M, E1, E2
            # One degradation draw per timestep pair: same randomness for both D calls within a step.
            th = torch.full((B,), t_hi, dtype=torch.long, device=dev)
            tl = torch.full((B,), t_lo, dtype=torch.long, device=dev)
            if mode == "resample":
                mm = torch.rand((B, H, W), device=dev); ee1 = torch.randn((B, H, W), device=dev); ee2 = torch.randn((B, H, W), device=dev)
                xh = diff.degrade(x0p, norms, ng, th, fixed_mask_rand=mm, fixed_e1=ee1, fixed_e2=ee2)
                xl = diff.degrade(x0p, norms, ng, tl, fixed_mask_rand=mm, fixed_e1=ee1, fixed_e2=ee2)
            else:
                xh = diff.degrade(x0p, norms, ng, th, fixed_mask_rand=m_, fixed_e1=e1_, fixed_e2=e2_)
                xl = diff.degrade(x0p, norms, ng, tl, fixed_mask_rand=m_, fixed_e1=e1_, fixed_e2=e2_)
            x = x - xh + xl
    return x


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="../../results/postmortem_3_3/cold_causes_valid.json")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = GalaxyDataset("../../data-full/")
    split, _ = load_split("../../results/splits/split_v1.json")
    idx = split["valid"][::max(1, len(split["valid"]) // a.n)][: a.n]
    dec = OldDecoder(timesteps=T).to(dev).eval()
    dec.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=True))
    diff = Diffusion(timesteps=T, max_sparsity=SP, max_noise_std=STD, device=dev)
    loader = DataLoader(Subset(ds, idx), batch_size=a.batch, shuffle=False)
    res = {"meta": {"ckpt": CKPT, "n": len(idx), "sparsity": SP, "noise_std": STD, "T": T,
                    "split": "valid (data-full, grouped split v1)", "torch": torch.__version__}}
    acc = {k: ([], []) for k in ["input", "direct", "matched", "mask_mismatch", "noise_mismatch", "resample"]}
    for bi, (x0, norms, ng) in enumerate(loader):
        x0, norms, ng = x0.to(dev), norms.to(dev), ng.to(dev)
        B, _, H, W = x0.shape
        g = torch.Generator(device=dev).manual_seed(100 + bi)
        M = torch.rand((B, H, W), device=dev, generator=g)
        E1 = torch.randn((B, H, W), device=dev, generator=g)
        E2 = torch.randn((B, H, W), device=dev, generator=g)
        M2 = torch.rand((B, H, W), device=dev, generator=g)
        E1b = torch.randn((B, H, W), device=dev, generator=g)
        E2b = torch.randn((B, H, W), device=dev, generator=g)
        with torch.no_grad():
            xT = diff.degrade(x0, norms, ng, torch.full((B,), T, device=dev, dtype=torch.long),
                                 fixed_mask_rand=M, fixed_e1=E1, fixed_e2=E2)
            preds = {
                "input": xT,
                "direct": dec(xT, torch.full((B,), T, device=dev, dtype=torch.long)),
                "matched": run_algo2(dec, diff, xT, norms, ng, 1000, "matched", M, E1, E2),
                "mask_mismatch": run_algo2(dec, diff, xT, norms, ng, 1000, "matched", M2, E1, E2),
                "noise_mismatch": run_algo2(dec, diff, xT, norms, ng, 1000, "matched", M, E1b, E2b),
                "resample": run_algo2(dec, diff, xT, norms, ng, 1000, "resample"),
            }
        for k, p in preds.items():
            m, r = per_sample_metrics(x0.cpu(), p.cpu())
            acc[k][0].append(m); acc[k][1].append(r)
        print("batch", bi, {k: round(float(np.mean(v[1])), 4) for k, v in acc.items()}, flush=True)
    res["conditions"] = {k: {"mse": float(np.concatenate(v[0]).mean()), "rho": float(np.concatenate(v[1]).mean())}
                         for k, v in acc.items()}
    print(json.dumps(res["conditions"], indent=1), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
