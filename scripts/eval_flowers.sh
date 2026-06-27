#!/usr/bin/env bash
set -e
export PAAE_CONFIG=${PAAE_CONFIG:-configs/swin-flowers.yaml}
export PAAE_DATA_ROOT=${PAAE_DATA_ROOT:-./data}
export PAAE_RESUME=${PAAE_RESUME:-checkpoints/PAAE_FLOWERS.pth}
export PAAE_EVAL_MODE=1
export PAAE_CUDA_VISIBLE=${PAAE_CUDA_VISIBLE:-0}
python -m main
