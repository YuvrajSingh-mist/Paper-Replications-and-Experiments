#!/usr/bin/env bash
# Run instruct-suite tasks as separate processes so each run can flush
# results/samples and free MPS RAM before the next.
# Uses existing ./scripts/evals.sh only — no new Python/CLI flags.
#
#   ./scripts/eval_instruct_split.sh HuggingFaceTB/SmolLM-360M-Instruct
#   ./scripts/eval_instruct_split.sh MODEL --limit 8
#   SMOKE=1 ./scripts/eval_instruct_split.sh MODEL
#
# One output folder per task. Per-sample answers (--log-samples) only for ifeval+gsm8k.
# Same batch_size=8 for every shard.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  sed -n '2,12p' "$0"
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
    extra=(--limit 8 "${extra[@]}")
  else
    extra=(--limit 8)
  fi
fi

safe_name="$(basename "${MODEL%/}" | tr '/:' '__')"
base_out="${OUT_BASE:-outputs/lm_eval/${safe_name}}"
mkdir -p "${base_out}"

BATCH="${BATCH:-8}"
# name|tasks|log_samples(0|1)|num_fewshot_or_-
shards=(
  "ifeval|ifeval|1|-"
  "hellaswag|hellaswag|0|-"
  "arc_easy|arc_easy|0|-"
  "arc_challenge|arc_challenge|0|-"
  "piqa|piqa|0|-"
  "gsm8k|gsm8k|1|1"
)

echo "==> split instruct eval (lm-eval)"
echo "    model:    ${MODEL}"
echo "    base:     ${base_out}"
echo "    extra:    ${extra[*]:-(none)}"
echo "    batch:    ${BATCH}"
echo "    details:  ifeval + gsm8k only"

for spec in "${shards[@]}"; do
  IFS='|' read -r name tasks log_flag few <<< "${spec}"
  out="${base_out}/${name}"
  echo
  echo "======== shard=${name}  TASKS=${tasks}  OUT=${out}  batch=${BATCH}  log_samples=${log_flag} ========"
  mkdir -p "${out}"
  flags=(--batch-size "${BATCH}" --tasks "${tasks}")
  if [[ ${#extra[@]} -gt 0 ]]; then
    flags+=("${extra[@]}")
  fi
  if [[ "${log_flag}" == "1" ]]; then
    flags+=(--log-samples)
  fi
  if [[ "${few}" != "-" ]]; then
    flags+=(--num-fewshot "${few}")
  fi
  OUT="${out}" ./scripts/evals.sh "${MODEL}" "${flags[@]}"
  echo "======== shard=${name} done ========"
done

echo
echo "==> all shards finished under ${base_out}/"
