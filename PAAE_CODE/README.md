# PAAE: Part-Aware Adaptive Alignment and Enhancement for Fine-Grained Visual Classification

This repository contains the PyTorch implementation of **Part-Aware Adaptive Alignment and Enhancement (PAAE)** for fine-grained visual classification.

PAAE consists of three main components:

- **Progressive Part Mining (PPMiner)** for hierarchical part-level cue mining.
- **Adaptive Scale Displacement Alignment (ADAM)** for multi-scale feature alignment.
- **Dual-Path Feature Enhancement (DFEM)** for channel-spatial enhancement and attention diversity regularization.

The repository is organized to support reproducible training, evaluation, and configuration tracking.

## Repository structure

```text
PAAE/
├── configs/                  # Dataset/model configuration files
├── docs/                     # Dataset preparation and reproducibility instructions
├── models/                   # Backbone and PAAE model definitions
├── pretrained/               # Swin pretrained weights and released PAAE weights (not committed)
├── scripts/                  # One-command training/evaluation scripts
├── settings/                 # Configuration loading and experiment setup
├── utils/                    # Datasets, dataloaders, optimizer, scheduler, evaluation utilities
├── main.py                   # Main training/evaluation entry
├── setup.py                  # Runtime configuration entry
├── requirements.txt
└── environment.yml
```

## Installation

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate paae
```

### Option 2: pip

```bash
conda create -n paae python=3.9 -y
conda activate paae
pip install -r requirements.txt
```

## Data preparation

Raw datasets are **not redistributed** in this repository. Please download them from the official sources and organize them as described in [`docs/DATASETS.md`](docs/DATASETS.md).

Expected root structure:

```text
$PAAE_DATA_ROOT/
├── CUB_200_2011/
│   └── CUB_200_2011/
│       ├── images/
│       ├── images.txt
│       ├── image_class_labels.txt
│       └── train_test_split.txt
├── dogs/
│   ├── Images/
│   ├── Annotation/
│   ├── train_list.mat
│   └── test_list.mat
└── flowers/
    ├── jpg/
    ├── imagelabels.mat
    └── setid.mat
```

The code follows the official CUB-200-2011 train/test split and the standard Stanford Dogs split files. For Oxford 102 Flowers, run `python prepare_flowers.py --root $PAAE_DATA_ROOT/flowers` once to generate `train.txt` / `test.txt` (see [`docs/DATASETS.md`](docs/DATASETS.md)).

## Pretrained weights and checkpoints

Please put Swin Transformer pretrained weights and released PAAE checkpoints under `pretrained/` or `checkpoints/`.
See [`pretrained/README.md`](pretrained/README.md) and [`checkpoints/README.md`](checkpoints/README.md) for the expected file names.

Before final submission, we recommend uploading released weights to **GitHub Releases**, **Zenodo**, **Google Drive**, **OneDrive**, or another stable storage service, and then linking them from `checkpoints/README.md`.

## Training

Set the dataset root and run one of the scripts:

```bash
export PAAE_DATA_ROOT=/path/to/datasets
bash scripts/train_cub.sh
bash scripts/train_dogs.sh
bash scripts/train_flowers.sh
```

For Windows PowerShell, run:

```powershell
$env:PAAE_DATA_ROOT="D:\path\to\datasets"
python -m main
```

The default random seed is `1`. Configuration files are stored under `configs/`.

## Evaluation

After downloading or training a checkpoint, run:

```bash
export PAAE_DATA_ROOT=/path/to/datasets
export PAAE_RESUME=checkpoints/PAAE_CUB.pth
bash scripts/eval_cub.sh
```

For Stanford Dogs:

```bash
export PAAE_DATA_ROOT=/path/to/datasets
export PAAE_RESUME=checkpoints/PAAE_DOGS.pth
bash scripts/eval_dogs.sh
```

For Oxford 102 Flowers:

```bash
export PAAE_DATA_ROOT=/path/to/datasets
export PAAE_RESUME=checkpoints/PAAE_FLOWERS.pth
bash scripts/eval_flowers.sh
```

## Reproducibility notes

The repository includes:

- Source code
- Training scripts
- Evaluation scripts
- Dataset preparation instructions
- Configuration files
- Environment files
- Random seed setting
- Pretrained/checkpoint release instructions

Please refer to [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for a step-by-step reproduction guide.

## Expected results

| Dataset | Backbone | Input size | Top-1 Acc. |
|---|---:|---:|---:|
| CUB-200-2011 | Swin-Base | 384 | 92.6 |
| Stanford Dogs | Swin-Base | 384 | 95.1 |
| Oxford 102 Flowers | Swin-Base | 384 | 99.7 |

## Acknowledgement

This codebase uses common components from Swin Transformer training utilities and fine-grained recognition codebases. If any third-party code is used, please keep the original license and attribution notices.

## Citation

If this repository is useful for your research, please cite our paper after publication.

```bibtex
@article{PAAE2026,
  title={Part-Aware Adaptive Alignment and Enhancement for Fine-Grained Visual Classification},
  author={Zhao, Qianhao and Liu, Jianlei and Zhang, Ke},
  journal={The Visual Computer},
  year={2026}
}
```
