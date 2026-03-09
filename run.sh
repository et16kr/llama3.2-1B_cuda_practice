#!/bin/sh
set -eu

MODEL_DIR="${MODEL_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_TXT="${INPUT_TXT:-./data/requests.txt}"
OUTPUT_TXT="${OUTPUT_TXT:-./data/responses.txt}"
MAIN_BIN="${MAIN_BIN:-./main}"

if [ -z "${MODEL_DIR}" ]; then
  echo "MODEL_DIR is not set."
  echo "Example:"
  echo "  MODEL_DIR=/path/to/Llama-3.2-1B-Instruct INPUT_TXT=./data/requests.txt make run"
  exit 1
fi

"${PYTHON_BIN}" ./scripts/run_text_generation.py \
  --model-dir "${MODEL_DIR}" \
  --input "${INPUT_TXT}" \
  --output "${OUTPUT_TXT}" \
  --main-binary "${MAIN_BIN}"
