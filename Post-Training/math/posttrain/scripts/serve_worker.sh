#!/usr/bin/env bash
# Rollout Mac Mini — run after: ./setup.sh --worker
#   MODEL=HuggingFaceTB/SmolLM-360M-Instruct PORT=8080 bash scripts/serve_worker.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${MODEL:-HuggingFaceTB/SmolLM-360M-Instruct}"
PORT="${PORT:-8080}"
ADAPTER_DIR="${ADAPTER_DIR:-$HOME/grpo_adapters/default/grpo}"
LORA_NAME="${LORA_NAME:-grpo}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
GPU_MEM="${GPU_MEM:-0.9}"

mkdir -p "$ADAPTER_DIR"

if [[ -f "${HOME}/.venv-vllm-metal/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.venv-vllm-metal/bin/activate"
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm not on PATH — run: ${ROOT}/setup.sh --worker" >&2
  exit 1
fi

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1

echo "serving ${MODEL} on :${PORT} with LoRA '${LORA_NAME}' → ${ADAPTER_DIR}"
exec vllm serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --enable-lora \
  --lora-modules "${LORA_NAME}=${ADAPTER_DIR}" \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --gpu-memory-utilization "${GPU_MEM}"
