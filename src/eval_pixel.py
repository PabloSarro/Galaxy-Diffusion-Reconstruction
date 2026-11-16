"""Pixel-level evaluation (MSE, Pearson) of a trained decoder on the validation split.

Same split / degradation / metric code as train_decoder.py (evaluate_cold_diffusion);
only the paths and settings are CLI arguments.
"""
import argparse
import torch
from torch.utils.data import DataLoader

from dataset import GalaxyDataset
from model import Decoder
from degradation import Degradation
from helpers import set_seed, generate_train_valid_datasets, evaluate_cold_diffusion
from eval_args import add_eval_args


def main(argv=None):
    parser = argparse.ArgumentParser()
    add_eval_args(parser)
    parser.add_argument("--noise-std", type=float, required=True, help="Single sigma to evaluate at")
    parser.add_argument("--max-batches", type=int, default=20,
                        help="Batches of the validation loader (train_decoder.py used 20); 0 = all")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device} | torch {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"decoder={args.decoder_path} sparsity={args.sparsity} sigma={args.noise_std} seed={args.seed}")

    set_seed(args.seed)
    dataset = GalaxyDataset(data_folder=args.data_folder)
    _, valid_dataset = generate_train_valid_datasets(dataset, frac=0.8)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    degradation = Degradation(sparsity=args.sparsity, noise_std=args.noise_std, device=device)
    decoder = Decoder().to(device)
    evaluate_cold_diffusion(
        decoder=decoder, degradation=degradation, device=device, valid_loader=valid_loader,
        max_batches=args.max_batches or None, model_path=args.decoder_path,
    )


if __name__ == "__main__":
    main()
