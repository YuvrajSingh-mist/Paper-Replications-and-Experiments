#!/usr/bin/env bash
# Run LightEval: ./evals.sh <model_path_or_hf_id> [--limit N ...]
#
#   ./evals.sh HuggingFaceTB/SmolLM-360M-Instruct
#   ./evals.sh outputs/sft-gsm8k/final
#   SUITE=base ./evals.sh HuggingFaceTB/SmolLM-360M
#   SUITE=math ./evals.sh MODEL
#   SUITE=instruct_full ./evals.sh MODEL
#   ./evals.sh MODEL --limit 32
#   TASKS='gsm8k|5,ifeval|0' ./evals.sh MODEL
#   OUT=outputs/my-run ./evals.sh MODEL
#
# Suites (override with TASKS=...):
#   instruct (default) — ifeval, hellaswag, arc, piqa, gsm8k
#   instruct_full      — instruct + mmlu_pro + bigbench_hard|3
#   base               — hellaswag, arc, piqa, mmlu_pro, …
#   math               — gsm8k|5
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

[[ -d .venv ]] || { echo "missing .venv — run: ./setup.sh" >&2; exit 1; }

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  sed -n '2,18p' "$0"
  exit 0
fi

MODEL="$1"
shift

SUITE="${SUITE:-instruct}"
# Output folder = model basename (model-specific only here).
safe_name="$(basename "${MODEL%/}" | tr '/:' '__')"
OUT="${OUT:-outputs/lighteval/${safe_name}}"

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

exec uv run math-eval \
  --config configs/eval.yaml \
  --model "${MODEL}" \
  --output-path "${OUT}" \
  "${extra[@]}" \
  "$@"
