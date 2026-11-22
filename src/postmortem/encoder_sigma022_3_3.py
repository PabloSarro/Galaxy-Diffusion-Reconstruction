"""Latent DDPM, step 1: encoder trained AT sigma = 0.22 (never reuse a 0.01 encoder).

Same design as the archived train_encoder.py (5b7efd0): z_true = Encoder(clean map),
z_obs = Encoder(degraded map), InfoNCE between them. Degradation: sparsity 0.7, sigma 0.22
(the same observation model as the other sigma=0.22 runs). Split: grouped split v1.
Resumable (last.pt each epoch); best.pt by validation loss.
"""
import argparse, importlib.util, json, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(HERE, ".."))
from dataset import GalaxyDataset
from degradation import Degradation
from splits import load_split
from helpers import set_seed


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


LAT = _load("old_latent_model", os.path.join(ROOT, "archive", "diffusion_7b9b37c", "model_latent_5b7efd0.py"))


def main():
    ap = argparse.ArgumentParser()
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
    enc = LAT.Encoder(latent_dim=a.latent_dim).to(dev)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-4)
    os.makedirs(a.out, exist_ok=True)
    last, best_path = os.path.join(a.out, "last.pt"), os.path.join(a.out, "encoder_best.pt")
    start, best, hist = 0, float("inf"), []
    if os.path.isfile(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        enc.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start, best, hist = ck["epoch"], ck["best"], ck["hist"]
        g.set_state(ck["gen"]); torch.set_rng_state(ck["rng"])
        print(f"resumed after {start} epochs", flush=True)
    torch.manual_seed(a.seed)
    for ep in range(start, a.epochs):
        enc.train(); tot = 0.0
        for x0, n, ng in tl:
            x0, n, ng = x0.to(dev), n.to(dev), ng.to(dev)
            opt.zero_grad()
            ns, _ = deg.degrade(x0, n, ng)
            loss = LAT.contrastive_loss(enc(x0), enc(ns))
            loss.backward(); opt.step(); tot += loss.item()
        enc.eval(); vtot = 0.0
        with torch.no_grad():
            torch.manual_seed(1000 + ep)  # fixed degradation draws for validation
            for x0, n, ng in vl:
                x0, n, ng = x0.to(dev), n.to(dev), ng.to(dev)
                ns, _ = deg.degrade(x0, n, ng)
                vtot += LAT.contrastive_loss(enc(x0), enc(ns)).item()
        tr, va = tot / len(tl), vtot / len(vl)
        hist.append([tr, va])
        if va < best:
            best = va; torch.save(enc.state_dict(), best_path)
        torch.save({"model": enc.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "best": best,
                    "hist": hist, "gen": g.get_state(), "rng": torch.get_rng_state()}, last)
        print(f"epoch {ep+1}/{a.epochs} train {tr:.5f} valid {va:.5f} best {best:.5f}", flush=True)
    json.dump({"noise_std": a.noise_std, "sparsity": a.sparsity, "epochs": a.epochs, "best_valid": best, "hist": hist},
              open(os.path.join(a.out, "encoder_result.json"), "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
