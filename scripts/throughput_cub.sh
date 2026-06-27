#!/usr/bin/env bash
set -e
export PAAE_CONFIG=${PAAE_CONFIG:-configs/swin-cub.yaml}
export PAAE_DATA_ROOT=${PAAE_DATA_ROOT:-./data}
export PAAE_RESUME=${PAAE_RESUME:-checkpoints/PAAE_CUB.pth}
export PAAE_THROUGHPUT=1
export PAAE_CUDA_VISIBLE=${PAAE_CUDA_VISIBLE:-0}
python -m main
