#!/usr/bin/env bash

MODEL=${MODEL:-"Qwen/Qwen3-0.6B"}
DTYPE=${DTYPE:-"float16"}
GPU_UTIL=${GPU_UTIL:-"0.8"}
MAX_LEN=${MAX_LEN:-"10240"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8000"}
EXTRA_ARGS=${EXTRA_ARGS:-""}

echo "Starting vLLM locally with model: $MODEL"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --dtype "$DTYPE" \
  --gpu_memory_utilization "$GPU_UTIL" \
  --max_model_len "$MAX_LEN" \
  --host "$HOST" \
  --port "$PORT"
  $EXTRA_ARGS
