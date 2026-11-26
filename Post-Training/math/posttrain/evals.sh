#!/usr/bin/env bash
# Run LightEval: ./evals.sh <model_path_or_hf_id> [extra flags…]
#
#   ./evals.sh HuggingFaceTB/SmolLM-360M-Instruct
#   SUITE=math ./evals.sh MODEL              # gsm8k|5 only
#   SUITE=instruct_full ./evals.sh MODEL
#   ./evals.sh MODEL --limit 8 --save-details
#   TASKS='gsm8k|5,ifeval|0' ./evals.sh MODEL
#   OUT=outputs/lighteval/my-run ./evals.sh MODEL
#
# Suites (override with TASKS=...): instruct | instruct_full | base | math
# Split full instruct (avoids end-of-suite OOM): ./scripts/eval_instruct_split.sh MODEL
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

[[ -d .venv ]] || { echo "missing .venv — run: ./setup.sh" >&2; exit 1; }

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  sed -n '2,14p' "$0"
  exit 0
fi

MODEL="$1"
shift

SUITE="${SUITE:-instruct}"
# Output folder = model basename only (experiment identity lives in the git branch).
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

exec uv run eval \
  --config configs/eval.yaml \
  --model "${MODEL}" \
  --output-path "${OUT}" \
  "${extra[@]}" \
  "$@"
