"""Task 3.3: latent-DDPM causes, using the archived 5b7efd0 encoder and the data-full VALID split.

1. Conditioning (code): the latent decoder takes z = Encoder(observation) only; the spatial
   observation is not an input (archived train_decoder.py: decoder(x_t, z_obs, t)).
2. InfoNCE / instance identity: linear probe for class from z, and nearest-neighbour retrieval:
   how often is z's nearest neighbour the same class, and the same simulation halo/cluster?
   Observation = degraded valid maps (sparsity 0.25, sigma 0.01, as in the old encoder training).
3. Data scale: std of x0 vs the unit-variance prior. SNR(t) = alpha_bar * var(x0) / (1 - alpha_bar)
   with the archived linear schedule (beta 1e-4 to 2e-2, T=1000).
"""
import importlib.util, json, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(HERE, ".."))
from dataset import GalaxyDataset
from degradation import Degradation
from splits import load_split


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


enc_mod = _load("old_latent_model", os.path.join(ROOT, "archive", "diffusion_7b9b37c", "model_latent_5b7efd0.py"))
CKPT = os.environ.get("LATENT_ENCODER", os.path.join(ROOT, "archive", "old_results", "encoder", "dim128_0.01_0.25", "encoder_best.pt"))


NOISE = float(os.environ.get("LATENT_NOISE", "0.01"))  # probe observation noise (encoder trained at 0.01)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    ds = GalaxyDataset(os.path.join(ROOT, "data-full"))
    split, _ = load_split(os.path.join(ROOT, "results", "splits", "split_v1.json"))
    enc = enc_mod.Encoder(latent_dim=128).to(dev).eval()
    enc.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=True))
    deg = Degradation(sparsity=0.25, noise_std=NOISE, device=dev)

    def embed(idx):
        zs = []
        torch.manual_seed(0)
        with torch.no_grad():
            for x, n, g in DataLoader(Subset(ds, idx), batch_size=64, shuffle=False):
                x, n, g = x.to(dev), n.to(dev), g.to(dev)
                ns, _ = deg.degrade(x, n, g)
                zs.append(enc(ns).cpu().numpy())
        return np.concatenate(zs)

    tr, va = split["train"], split["valid"]
    z_tr, z_va = embed(tr), embed(va)
    y_tr, y_va = ds.targets[tr], ds.targets[va]
    g_va = ds.groups[va]
    out = {"meta": {"encoder": CKPT, "sparsity": 0.25, "noise_std": NOISE, "split": "valid (grouped v1)"}}

    clf = LogisticRegression(max_iter=2000).fit(z_tr, y_tr)
    out["probe_class_acc_valid"] = float(clf.score(z_va, y_va))
    out["probe_class_chance_majority"] = float(np.bincount(y_va).max() / len(y_va))

    zn = z_va / (np.linalg.norm(z_va, axis=1, keepdims=True) + 1e-12)
    sim = zn @ zn.T
    np.fill_diagonal(sim, -np.inf)
    nn = sim.argmax(axis=1)
    out["nn_same_class"] = float((y_va[nn] == y_va).mean())
    out["nn_same_group"] = float((g_va[nn] == g_va).mean())
    out["nn_same_group_chance"] = float(sum((g_va == g).sum() ** 2 - (g_va == g).sum() for g in np.unique(g_va)) / (len(g_va) * (len(g_va) - 1)))

    # 3. Data scale vs the unit-variance prior (archived schedule).
    x_all = ds.images[va]
    var = float(np.var(x_all))
    beta = np.linspace(1e-4, 2e-2, 1000)
    abar = np.cumprod(1 - beta)
    snr = lambda v: abar * v / (1 - abar)
    out["data_std"] = float(np.sqrt(var))
    out["frac_t_snr_below_1_unit_var"] = float((snr(1.0) < 1).mean())
    out["frac_t_snr_below_1_data_var"] = float((snr(var) < 1).mean())
    out["t_snr_eq_1_data_var"] = int(np.argmax(snr(var) < 1))

    os.makedirs(os.path.join(ROOT, "results", "postmortem_3_3"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "postmortem_3_3", f"latent_causes_valid_noise{NOISE}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
