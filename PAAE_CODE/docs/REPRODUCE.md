# Reproduction guide

This document summarizes the steps needed to reproduce the main results.

## 1. Environment

```bash
conda env create -f environment.yml
conda activate paae
```

or

```bash
pip install -r requirements.txt
```

## 2. Dataset

Prepare CUB-200-2011, Stanford Dogs, and Oxford 102 Flowers following `docs/DATASETS.md`. For Oxford 102 Flowers, also run `python prepare_flowers.py --root $PAAE_DATA_ROOT/flowers` once to generate `train.txt` / `test.txt`.

```bash
export PAAE_DATA_ROOT=/path/to/datasets
```

## 3. Backbone pretrained weights

Put the Swin pretrained model in `pretrained/`:

```text
pretrained/Swin Base.pth
pretrained/Swin Base 1k.pth
```

## 4. Training

```bash
bash scripts/train_cub.sh
bash scripts/train_dogs.sh
bash scripts/train_flowers.sh
```

## 5. Evaluation

```bash
export PAAE_RESUME=checkpoints/PAAE_CUB.pth
bash scripts/eval_cub.sh

export PAAE_RESUME=checkpoints/PAAE_DOGS.pth
bash scripts/eval_dogs.sh

export PAAE_RESUME=checkpoints/PAAE_FLOWERS.pth
bash scripts/eval_flowers.sh
```

## 6. Random seed and configuration

The default random seed is `42`, defined in `settings/defaults.py`.
The training configuration is saved automatically to the experiment output directory.

Important settings:

- Backbone: Swin Transformer Base
- Input size: 384 x 384
- Optimizer: AdamW
- Weight decay: 1e-8
- Warm-up: 5 epochs
- AMP: enabled by default

## 7. Notes

Due to hardware and software differences, minor numerical variations may appear. Please report Python/PyTorch/CUDA versions when comparing reproduced results.
