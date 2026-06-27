@echo off
if "%PAAE_CONFIG%"=="" set PAAE_CONFIG=configs/swin-cub.yaml
if "%PAAE_DATA_ROOT%"=="" set PAAE_DATA_ROOT=.
if "%PAAE_RESUME%"=="" set PAAE_RESUME=checkpoints/PAAE_CUB.pth
set PAAE_EVAL_MODE=1
if "%PAAE_CUDA_VISIBLE%"=="" set PAAE_CUDA_VISIBLE=0
python -m main
