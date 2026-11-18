"""Grouped, class-stratified train / valid / test split (task 3.1, decision D4).

Every sample of one group (BAHAMAS cluster across all sigma files, or one DARKSKIES halo)
goes to the same split. Groups are assigned greedily so that each class's share in every
split is close to the target fraction. The result is saved as a JSON index file so that
every later run uses the same indices.
"""
import json
import os

import numpy as np

SPLIT_NAMES = ("train", "valid", "test")


def make_grouped_split(groups, targets, fractions=(0.70, 0.15, 0.15), seed=0):
    """Return {'train': idx, 'valid': idx, 'test': idx} (sorted int arrays)."""
    groups = np.asarray(groups)
    targets = np.asarray(targets)
    assert len(groups) == len(targets)
    assert abs(sum(fractions) - 1.0) < 1e-9
    classes = np.unique(targets)
    class_total = np.array([(targets == c).sum() for c in classes], dtype=float)
    target_counts = np.outer(np.asarray(fractions), class_total)  # (3 splits, n_classes)

    # Per-group class counts.
    uniq, inv = np.unique(groups, return_inverse=True)
    group_counts = np.zeros((len(uniq), len(classes)))
    for g_idx in range(len(uniq)):
        members = np.where(inv == g_idx)[0]
        for c_idx, c in enumerate(classes):
            group_counts[g_idx, c_idx] = (targets[members] == c).sum()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    # Large groups first (classic greedy bin packing), random order within equal sizes.
    order = sorted(order, key=lambda i: -group_counts[i].sum())

    assigned = np.zeros((len(SPLIT_NAMES), len(classes)))
    split_of_group = np.empty(len(uniq), dtype=int)
    for g in order:
        best, best_cost = None, None
        for s in range(len(SPLIT_NAMES)):
            after = assigned[s] + group_counts[g]
            # Cost: how far this split's class counts are above their targets (relative).
            cost = np.sum(np.maximum(after - target_counts[s], 0) / np.maximum(target_counts[s], 1))
            if best_cost is None or cost < best_cost - 1e-12:
                best, best_cost = s, cost
        split_of_group[g] = best
        assigned[best] += group_counts[g]

    sample_split = split_of_group[inv]
    return {name: np.where(sample_split == s)[0] for s, name in enumerate(SPLIT_NAMES)}


def class_report(split, targets):
    targets = np.asarray(targets)
    out = {}
    for name, idx in split.items():
        vals, counts = np.unique(targets[idx], return_counts=True)
        out[name] = {int(v): int(c) for v, c in zip(vals, counts)}
    return out


def save_split(path, split, meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"meta": meta, **{k: [int(i) for i in v] for k, v in split.items()}}
    with open(path, "w") as f:
        json.dump(payload, f)


def load_split(path):
    with open(path) as f:
        payload = json.load(f)
    return {k: np.asarray(payload[k], dtype=int) for k in SPLIT_NAMES}, payload["meta"]
