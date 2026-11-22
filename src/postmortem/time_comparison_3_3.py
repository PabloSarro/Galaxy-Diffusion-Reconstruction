"""Task 3.3 (D12): time comparison for cold diffusion on data-full, s=0.7, 2 seeds.

Modes (one change each, archived 7b9b37c model and degradation):
  random  t ~ U{1..T} per sample; degrade at t; decoder sees t (the old cold-diffusion recipe)
  fixed   always t = T; degrade at T; decoder sees t = T
  none    always degrade at T, but the decoder gets no time information (t fed as 0 everywhere)
Training: archived batch size 16, AdamW lr 1e-4, weight decay 1e-4 (as the archived train_decoder.py),
epochs as given. Split: grouped split v1 (train/valid). Checkpoints: last.pt every epoch (resume),
best.pt by valid loss.  Evaluation after training: direct prediction xT -> x0 at t=T on VALID.
"""
import argparse, importlib.util, json, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(HERE, ".."))
from dataset import GalaxyDataset
from splits import load_split
from eval_split import per_sample_metrics
from helpers import set_seed


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


ARCH = os.path.join(ROOT, "archive", "diffusion_7b9b37c")
Diffusion = _load("old_diffusion", os.path.join(ARCH, "diffusion.py")).Diffusion
OldDecoder = _load("old_model", os.path.join(ARCH, "model.py")).Decoder
T = 1000


def batch_t(mode, B, dev):
    if mode == "random":
        return torch.randint(1, T + 1, (B,), device=dev)
    return torch.full((B,), T, device=dev, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["random", "fixed", "none"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--sparsity", type=float, default=0.70)
    ap.add_argument("--noise-std", type=float, default=0.005)
    ap.add_argument("--out", required=True)
    ap.add_argument("--maskin", action="store_true", help="Decoder also gets the observed-pixel mask as a 3rd input channel")
    ap.add_argument("--split-file", default=os.path.join(ROOT, "results", "splits", "split_v1.json"))
    a = ap.parse_args()
    set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = GalaxyDataset(os.path.join(ROOT, "data-full"))
    split, _ = load_split(a.split_file)
    g = torch.Generator().manual_seed(a.seed)
    tl = DataLoader(Subset(ds, split["train"]), batch_size=16, shuffle=True, generator=g)
    vl = DataLoader(Subset(ds, split["valid"]), batch_size=16, shuffle=False)
    dec = make_decoder(a.maskin).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=1e-4, weight_decay=1e-4)
    diff = Diffusion(timesteps=T, max_sparsity=a.sparsity, max_noise_std=a.noise_std, device=dev)
    os.makedirs(a.out, exist_ok=True)
    last, best_path = os.path.join(a.out, "last.pt"), os.path.join(a.out, "best.pt")
    start, best, hist = 0, float("inf"), []
    if os.path.isfile(last):  # resume
        ck = torch.load(last, map_location=dev, weights_only=False)
        dec.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start, best, hist = ck["epoch"], ck["best"], ck["hist"]
        g.set_state(ck["gen"]); torch.set_rng_state(ck["rng"])
        print(f"resumed after {start} epochs", flush=True)

    for ep in range(start, a.epochs):
        dec.train(); tot = 0.0
        for x0, n, ng in tl:
            x0, n, ng = x0.to(dev), n.to(dev), ng.to(dev)
            opt.zero_grad()
            loss = _loss_with_t(dec, diff, a, x0, n, ng, dev)
            loss.backward(); torch.nn.utils.clip_grad_norm_(dec.parameters(), 5.0); opt.step()
            tot += loss.item()
        dec.eval(); vtot = 0.0
        with torch.no_grad():
            for x0, n, ng in vl:
                x0, n, ng = x0.to(dev), n.to(dev), ng.to(dev)
                vtot += _loss_with_t(dec, diff, a, x0, n, ng, dev).item()
        tr, va = tot / len(tl), vtot / len(vl)
        hist.append([tr, va])
        if va < best:
            best = va; torch.save(dec.state_dict(), best_path)
        torch.save({"model": dec.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "best": best,
                    "hist": hist, "gen": g.get_state(), "rng": torch.get_rng_state()}, last)
        print(f"epoch {ep+1}/{a.epochs} train {tr:.6f} valid {va:.6f} best {best:.6f}", flush=True)

    if a.maskin:  # evaluated by eval_maskin_3_3.py (needs the observed mask)
        print("DONE (maskin: evaluate with eval_maskin_3_3.py)", flush=True)
        return
    # Evaluation: direct prediction at t=T on VALID (decoder gets t=T; for 'none' it gets 0).
    dec.load_state_dict(torch.load(best_path, map_location=dev, weights_only=True)); dec.eval()
    mse, rho = [], []
    torch.manual_seed(0)
    with torch.no_grad():
        for x0, n, ng in vl:
            x0, n, ng = x0.to(dev), n.to(dev), ng.to(dev)
            xT = diff.degrade(x0, n, ng, torch.full((x0.size(0),), T, device=dev, dtype=torch.long))
            tin = torch.zeros((x0.size(0),), device=dev, dtype=torch.long) if a.mode == "none" else torch.full((x0.size(0),), T, device=dev, dtype=torch.long)
            m, r = per_sample_metrics(x0.cpu(), dec(xT, tin).cpu())
            mse.append(m); rho.append(r)
    res = {"mode": a.mode, "seed": a.seed, "epochs": a.epochs, "sparsity": a.sparsity, "noise_std": a.noise_std,
           "valid_direct_mse": float(np.concatenate(mse).mean()), "valid_direct_rho": float(np.concatenate(rho).mean()),
           "best_valid_loss": best, "hist": hist}
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=2)
    print("RESULT", json.dumps({k: res[k] for k in ("mode", "seed", "valid_direct_mse", "valid_direct_rho")}), flush=True)


def make_decoder(maskin):
    """Archived decoder; with maskin the input has 3 channels (xT, observed mask)."""
    if not maskin:
        return OldDecoder(timesteps=T)
    return OldDecoder(timesteps=T, in_channels=3)  # final skip uses init_conv features, so no other change


def observed_mask(t, ng, dev, sparsity, mr):
    """Known pixels: not removed by the sparsity mask, and with galaxies (both observable)."""
    sp_t = (t.float() / T * sparsity).view(-1, 1, 1)
    removed = mr < sp_t
    return ((~removed) & (ng > 0)).unsqueeze(1).float()


def _loss_with_t(dec, diff, a, x0, n, ng, dev):
    B = x0.size(0)
    t = batch_t(a.mode, B, dev)  # 'none' and 'fixed' degrade at T; 'random' at sampled t
    if a.maskin:
        mr = torch.rand((B,) + x0.shape[2:], device=dev)
        xt = diff.degrade(x0, n, ng, t, fixed_mask_rand=mr)
        inp = torch.cat([xt, observed_mask(t, ng, dev, a.sparsity, mr)], dim=1)
        return ((dec(inp, t) - x0) ** 2).mean()
    xt = diff.degrade(x0, n, ng, t)
    t_dec = torch.zeros_like(t) if a.mode == "none" else t
    return ((dec(xt, t_dec) - x0) ** 2).mean()


if __name__ == "__main__":
    main()
