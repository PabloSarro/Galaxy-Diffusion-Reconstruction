"""Decoder training loop with resumable checkpoints (task 2.2).

Files written to `out_dir`:
  last.pt          full state after every epoch (model, optimizer, losses, all RNG states)
  decoder_best.pt  weights of the best validation epoch (same name as before, used by evaluation)

Writes are atomic (temp file + os.replace), so a job killed mid-write keeps the previous file.
With resume=True and last.pt present, training continues from the next epoch and reproduces the
uninterrupted run exactly (tests/test_resume.py).
"""
import os
import random

import numpy as np
import torch

from model import decoder_loss

LAST_NAME = "last.pt"
BEST_NAME = "decoder_best.pt"


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state():
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_last(path, decoder, optimizer, epoch, best_loss, train_losses, valid_losses, loader_gen):
    _atomic_save({
        "epoch": epoch,               # number of completed epochs
        "model": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "rng": _rng_state(),
        "loader_gen": loader_gen.get_state(),
    }, path)


def run_training(decoder, optimizer, degradation, train_loader, valid_loader, loader_gen,
                 epochs, out_dir, loss_method="MSE", alpha=0.0, device="cpu",
                 resume=False, log=print):
    """Train `decoder` up to `epochs` total epochs. Returns (train_losses, valid_losses, best_loss)."""
    last_path = os.path.join(out_dir, LAST_NAME)
    best_path = os.path.join(out_dir, BEST_NAME)
    os.makedirs(out_dir, exist_ok=True)

    start_epoch, best_loss, train_losses, valid_losses = 0, float("inf"), [], []
    if resume and os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        decoder.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        best_loss = ckpt["best_loss"]
        train_losses, valid_losses = ckpt["train_losses"], ckpt["valid_losses"]
        loader_gen.set_state(ckpt["loader_gen"])
        _set_rng_state(ckpt["rng"])
        start_epoch = ckpt["epoch"]
        log(f"Resumed from {last_path} after {start_epoch} completed epochs.")

    for epoch in range(start_epoch, epochs):
        total_train = 0.0
        decoder.train()
        for x0, norms, num_gals in train_loader:
            x0 = x0.to(device, non_blocking=True)
            norms = norms.to(device, non_blocking=True)
            num_gals = num_gals.to(device, non_blocking=True)

            optimizer.zero_grad()
            x_noised, _ = degradation.degrade(x0, norms, num_gals)
            x0_pred = decoder(x_noised)
            loss = decoder_loss(x0_pred, x0, method=loss_method, alpha=alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
            optimizer.step()
            total_train += loss.item()
        train_losses.append(total_train / len(train_loader))

        decoder.eval()
        total_valid = 0.0
        with torch.no_grad():
            for x0, norms, num_gals in valid_loader:
                x0 = x0.to(device, non_blocking=True)
                norms = norms.to(device, non_blocking=True)
                num_gals = num_gals.to(device, non_blocking=True)
                x_t, _ = degradation.degrade(x0, norms, num_gals)
                x0_pred = decoder(x_t)
                total_valid += decoder_loss(x0_pred, x0, method=loss_method, alpha=alpha).item()
        valid_losses.append(total_valid / len(valid_loader))
        decoder.train()

        log(f"Epoch {epoch+1}/{epochs} --> Train: {train_losses[-1]:.6f} | Valid: {valid_losses[-1]:.6f}.")
        if valid_losses[-1] < best_loss:
            best_loss = valid_losses[-1]
            _atomic_save(decoder.state_dict(), best_path)
            log("   Best model was stored")
        save_last(last_path, decoder, optimizer, epoch + 1, best_loss,
                  train_losses, valid_losses, loader_gen)

    return train_losses, valid_losses, best_loss
