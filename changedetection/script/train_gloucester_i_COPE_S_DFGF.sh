#!/usr/bin/env bash
# Gloucester I — COPE-S + DFGF (latest framework)
set -euo pipefail

ROOT="/root/autodl-tmp"
DATA="${ROOT}/data/Gloucester I"
PROJECT="${ROOT}/MambaCD"
LOG_DIR="${PROJECT}/changedetection/logs"
mkdir -p "${LOG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_Gloucester_I_COPE-S_DFGF_${STAMP}.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python "${PROJECT}/changedetection/script/train_MambaBCD_spatial_COPE_S_DFGF.py" \
  --dataset "Gloucester_I" \
  --train_dataset_path "${DATA}/train" \
  --train_data_list_path "${DATA}/train_list.txt" \
  --test_dataset_path "${DATA}/test" \
  --test_data_list_path "${DATA}/test_list.txt" \
  --crop_size 256 \
  --batch_size 8 \
  --max_iters 300000 \
  --learning_rate 1e-4 \
  --model_type "LEM-CD_COPE-S_DFGF_Gloucester_I_batch8_iter300000" \
  --model_param_path "${PROJECT}/changedetection/saved_models" \
  --pretrained_weight_path "${PROJECT}/pretrained_weight/spatialmamba_small_224_1k.pth" \
  --use-dfgf \
  --dfgf-frequency-gate \
  --dfgf-direction-selector \
  --use-cope-s \
  --cope-change-guide \
  --cope-direction-reweight \
  --cope-residual \
  --log_path "${LOG_FILE}" \
  "$@" \
  2>&1 | tee -a "${LOG_FILE}"
