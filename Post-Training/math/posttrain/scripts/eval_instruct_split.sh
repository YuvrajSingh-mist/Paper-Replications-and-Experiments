#!/usr/bin/env bash
# Run instruct-suite tasks as separate processes so each run can flush
# results/details and free MPS RAM before the next (avoids end-of-suite OOM).
# Uses existing ./evals.sh only — no new Python/CLI flags.
#
#   ./scripts/eval_instruct_split.sh HuggingFaceTB/SmolLM-360M-Instruct
#   ./scripts/eval_instruct_split.sh MODEL --limit 8 --save-details
#   SMOKE=1 ./scripts/eval_instruct_split.sh MODEL   # --limit 8 --save-details
#   PER_TASK=1 SMOKE=1 ./scripts/eval_instruct_split.sh MODEL  # one process per task
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  sed -n '2,11p' "$0"
  exit 0
fi

MODEL="$1"
shift

extra=()
if [[ $# -gt 0 ]]; then
  extra=("$@")
fi
if [[ "${SMOKE:-0}" == "1" ]]; then
  if [[ ${#extra[@]} -gt 0 ]]; then
    extra=(--limit 8 --save-details "${extra[@]}")
  else
    extra=(--limit 8 --save-details)
  fi
fi

safe_name="$(basename "${MODEL%/}" | tr '/:' '__')"
base_out="${OUT_BASE:-outputs/lighteval/${safe_name}}"
mkdir -p "${base_out}"

if [[ "${PER_TASK:-0}" == "1" ]]; then
  # Smoke / isolate every benchmark in its own process.
  shards=(
    "ifeval|ifeval|0"
    "hellaswag|hellaswag|0"
    "arc_easy|arc:easy|0"
    "arc_challenge|arc:challenge|0"
    "piqa|piqa_hf|0"
    "gsm8k|gsm8k|5"
  )
else
  # Full-run grouping: biggest alone; generative pair; small MC together.
  shards=(
    "hellaswag|hellaswag|0"
    "gen|ifeval|0,gsm8k|5"
    "mc|arc:easy|0,arc:challenge|0,piqa_hf|0"
  )
fi

echo "==> split instruct eval"
echo "    model:    ${MODEL}"
echo "    base:     ${base_out}"
echo "    per_task: ${PER_TASK:-0}"
echo "    extra:    ${extra[*]:-(none)}"

for spec in "${shards[@]}"; do
  name="${spec%%|*}"
  tasks="${spec#*|}"
  out="${base_out}/${name}"
  echo
  echo "======== shard=${name}  TASKS=${tasks}  OUT=${out} ========"
  mkdir -p "${out}"
  if [[ ${#extra[@]} -gt 0 ]]; then
    OUT="${out}" TASKS="${tasks}" ./evals.sh "${MODEL}" "${extra[@]}"
  else
    OUT="${out}" TASKS="${tasks}" ./evals.sh "${MODEL}"
  fi
  echo "======== shard=${name} done ========"
done

echo
echo "==> all shards finished under ${base_out}/"
