#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ===== 可调参数 =====
ENV_NAME="${ENV_NAME:-dd}"
TASK_NAME="${TASK_NAME:-test1}"
DATA_DIR="${DATA_DIR:-/home/hebu/dd_dataset}"
MODEL="${MODEL:-yolo11n-cls.pt}"

EPOCHS="${EPOCHS:-20}"
IMG_SIZE="${IMG_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR0="${LR0:-0.01}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"

EVAL_FREQ="${EVAL_FREQ:-2}"
NUM_EVAL="${NUM_EVAL:-100}"
SEED="${SEED:-42}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

python train.py \
  --data "$DATA_DIR" \
  --model "$MODEL" \
  --task-name "$TASK_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --imgsz "$IMG_SIZE" \
  --batch "$BATCH_SIZE" \
  --lr0 "$LR0" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --eval-freq "$EVAL_FREQ" \
  --num-eval "$NUM_EVAL" \
  --seed "$SEED" \
  --log-interval "$LOG_INTERVAL"
