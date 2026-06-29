#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
TEST_FILE="${TEST_FILE:-/path/to/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
RESTORE_FROM="${RESTORE_FROM:-/path/to/model}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29511}"
MERGE_SIZE="${MERGE_SIZE:-16}"
SEGMENT_SIZE="${SEGMENT_SIZE:-10000}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"


torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" ft_inference.py \
  --model_name_or_path "${MODEL_NAME}" \
  --test_file "${TEST_FILE}" \
  --restore_from "${RESTORE_FROM}" \
  --output_dir "${OUTPUT_DIR}" \
  --merge_size "${MERGE_SIZE}" \
  --segment_size "${SEGMENT_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --num_samples "${NUM_SAMPLES}"
