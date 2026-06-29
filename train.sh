#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
TRAIN_FILE="${TRAIN_FILE:-/path/to/train.jsonl}"
TEST_FILE="${TEST_FILE:-/path/to/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29502}"
MERGE_SIZE="${MERGE_SIZE:-16}"
SEGMENT_SIZE="${SEGMENT_SIZE:-30000}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"


torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" instruction_finetune.py \
  --model_name_or_path "${MODEL_NAME}" \
  --train_file "${TRAIN_FILE}" \
  --test_file "${TEST_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --train true \
  --merge_size "${MERGE_SIZE}" \
  --segment_size "${SEGMENT_SIZE}" \
  --max_steps "${MAX_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --debug_data True
  # --deepspeed ds_config.json \

