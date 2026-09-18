# Galaxy Diffusion Reconstruction

Reconstructs weak-lensing galaxy shear maps from sparse, noisy observations using a
U-Net decoder, and evaluates reconstructions against a kNN baseline via a classifier.

## Structure

- `src/dataset.py` — loads simulation data
- `src/degradation.py` — masking + shape-noise degradation
- `src/model.py` — decoder network
- `src/train_decoder.py` — trains the decoder
- `src/classification.py` — evaluates a decoder via a pretrained classifier
- `src/visualise_imgs.py` — visualizes reconstructions

## Requirements

`torch`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `netloader`

Data (`../data-full/`) and results (`../results/`) are expected as sibling directories.

## Usage

```bash
python src/train_decoder.py --loss MSE
python src/classification.py --decoder_path "../results/<experiment>/MSE/training/decoder_best.pt"
python src/visualise_imgs.py