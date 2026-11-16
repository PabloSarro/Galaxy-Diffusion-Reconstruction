"""Shared CLI arguments for the evaluation scripts (no logic, only paths and settings)."""
import argparse

DEFAULT_NOISE_STDS = [0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.22, 0.3]


def add_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--decoder-path", required=True, help="Path to decoder_best.pt")
    parser.add_argument("--output-dir", required=True, help="Folder for plots / logs")
    parser.add_argument("--data-folder", default="../data-full/")
    parser.add_argument("--sparsity", type=float, default=0.70)
    parser.add_argument("--noise-stds", type=float, nargs="+", default=DEFAULT_NOISE_STDS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser
