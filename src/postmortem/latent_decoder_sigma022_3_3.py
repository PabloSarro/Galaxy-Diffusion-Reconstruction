"""Latent DDPM, step 2: decoder on the sigma = 0.22 encoder latent (encoder frozen).

Archived recipe (5b7efd0 diffusion + model, old train_decoder.py): DDPM on the clean map x0, noise
prediction, decoder(x_t, z_obs, t) with z_obs = Encoder(degraded map); AdamW lr 1e-5, batch 64.
Degradation for z_obs: sparsity 0.7, sigma 0.22. Split: grouped v1. Resumable (last.pt).
After training: 1000-step DDPM sampling conditioned on z_obs for 64 VALID maps -> rho, MSE.
"""
import argparse, importlib.util, json, os, sys
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(HERE, ".."))
from dataset import GalaxyDataset
from degradation import Degradation
from splits import load_split
from eval_split import per_sample_metrics
from helpers import set_seed


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


ARCH = os.path.join(ROOT, "archive", "diffusion_7b9b37c")
LAT = _load("old_latent_model", os.path.join(ARCH, "model_latent_5b7efd0.py"))
DIFF = _load("old_latent_diffusion", os.path.join(ARCH, "diffusion_latent_5b7efd0.py"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--noise-std", type=float, default=0.22)
    ap.add_argument("--sparsity", type=float, default=0.70)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = GalaxyDataset(os.path.join(ROOT, "data-full"))
    split, _ = load_split(os.path.join(ROOT, "results", "splits", "split_v1.json"))
    g = torch.Generator().manual_seed(a.seed)
    tl = DataLoader(Subset(ds, split["train"]), batch_size=64, shuffle=True, generator=g)
    vl = DataLoader(Subset(ds, split["valid"]), batch_size=64, shuffle=False)
    deg = Degradation(sparsity=a.sparsity, noise_std=a.noise_std, device=dev)
    enc = LAT.Encoder(latent_dim=a.latent_dim).to(dev).eval()
    enc.load_state_dict(torch.load(a.encoder, map_location=dev, weights_only=True))
    for p in enc.parameters():
        p.requires_grad_(False)
    dec = LAT.Decoder(timesteps=1000, latent_dim=a.latent_dim).to(dev)
    diff = DIFF.Diffusion(timesteps=1000, device=dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=1e-5)
    os.makedirs(a.out, exist_ok=True)
    last, best_path = os.path.join(a.out, "last.pt"), os.path.join(a.out, "decoder_best.pt")
    start, best, hist = 0, float("inf"), []
    if os.path.isfile(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        dec.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start, best, hist = ck["epoch"], ck["best"], ck["hist"]
        g.set_state(ck["gen"]); torch.set_rng_state(ck["rng"])
        print(f"resumed after {start} epochs", flush=True)

    def loss_on(x0, n, ng):
        with torch.no_grad():
            ns, _ = deg.degrade(x0, n, ng)
            z = enc(ns)
        t = diff.sample_timesteps(x0.size(0))
        xt, noise = diff.q_sample(x0, t)
        return F.mse_loss(dec(xt, z, t), noise)

    for ep in range(start, a.epochs):
        dec.train(); tot = 0.0
        for x0, n, ng in tl:
            x0, n, ng = x0.to(dev), n.to(dev), ng.to(dev)
            opt.zero_grad()
            loss = loss_on(x0, n, ng)
            loss.backward(); torch.nn.utils.clip_grad_norm_(dec.parameters(), 5.0); opt.step()
            tot += loss.item()
        dec.eval(); vtot = 0.0
        with torch.no_grad():
            torch.manual_seed(1000 + ep)
            for x0, n, ng in vl:
                x0, n, ng = x0.to(dev), n.to(dev), ng.to(dev)
                vtot += loss_on(x0, n, ng).item()
        tr, va = tot / len(tl), vtot / len(vl)
        hist.append([tr, va])
        if va < best:
            best = va; torch.save(dec.state_dict(), best_path)
        torch.save({"model": dec.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "best": best,
                    "hist": hist, "gen": g.get_state(), "rng": torch.get_rng_state()}, last)
        print(f"epoch {ep+1}/{a.epochs} train {tr:.5f} valid {va:.5f} best {best:.5f}", flush=True)

    # Evaluation: DDPM sampling conditioned on z_obs, 64 VALID maps (same subset as the cold analyses).
    dec.load_state_dict(torch.load(best_path, map_location=dev, weights_only=True)); dec.eval()
    idx = split["valid"][:: max(1, len(split["valid"]) // 64)][:64]
    mse, rho = [], []
    torch.manual_seed(0)
    for x0, n, ng in DataLoader(Subset(ds, idx), batch_size=32, shuffle=False):
        x0, n, ng = x0.to(dev), n.to(dev), ng.to(dev)
        with torch.no_grad():
            ns, _ = deg.degrade(x0, n, ng)
            pred = diff.sample(dec, enc(ns), x0.shape)
        m, r = per_sample_metrics(x0.cpu(), pred.cpu())
        mse.append(m); rho.append(r)
    res = {"noise_std": a.noise_std, "sparsity": a.sparsity, "epochs": a.epochs, "encoder": a.encoder,
           "best_valid_loss": best, "valid_ddpm_mse": float(np.concatenate(mse).mean()),
           "valid_ddpm_rho": float(np.concatenate(rho).mean()), "n_eval": len(idx), "hist": hist}
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=2)
    print("RESULT", json.dumps({k: res[k] for k in ("valid_ddpm_mse", "valid_ddpm_rho")}), flush=True)


if __name__ == "__main__":
    main()
