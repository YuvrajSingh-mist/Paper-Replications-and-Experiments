#!/usr/bin/env bash
# Run lm-eval: ./scripts/evals.sh <model_path_or_hf_id> [extra flags…]
#
#   ./scripts/evals.sh HuggingFaceTB/SmolLM-360M-Instruct
#   SUITE=math ./scripts/evals.sh MODEL              # gsm8k only
#   ./scripts/evals.sh MODEL --limit 8 --log-samples
#   TASKS='gsm8k,ifeval' ./scripts/evals.sh MODEL
#   OUT=outputs/lm_eval/my-run ./scripts/evals.sh MODEL
#
# Suites (override with TASKS=...): instruct | math
# Split (details only on ifeval+gsm8k): ./scripts/eval_instruct_split.sh MODEL
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ -d .venv ]] || { echo "missing .venv — run: ./setup.sh" >&2; exit 1; }

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  sed -n '2,14p' "$0"
  exit 0
fi

MODEL="$1"
shift

SUITE="${SUITE:-instruct}"
safe_name="$(basename "${MODEL%/}" | tr '/:' '__')"
OUT="${OUT:-outputs/lm_eval/${safe_name}}"

export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "==> model:  ${MODEL}"
echo "==> suite:  ${SUITE}"
echo "==> output: ${OUT}"

extra=()
if [[ -n "${TASKS:-}" ]]; then
  extra+=(--tasks "${TASKS}")
else
  extra+=(--suite "${SUITE}")
fi

exec uv run eval \
  --config configs/eval.yaml \
  --model "${MODEL}" \
  --output-path "${OUT}" \
  "${extra[@]}" \
  "$@"
