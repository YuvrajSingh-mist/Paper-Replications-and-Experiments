#!/usr/bin/env bash
# One-shot setup for Apple Silicon Mac Minis.
#
#   ./setup.sh              # trainer box (uv + TRL/torch + lm-eval)
#   ./setup.sh --worker     # rollout box (vllm-metal) for GRPO
#   ./setup.sh --all        # both on the same machine
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="train"
for arg in "$@"; do
  case "$arg" in
    --worker) MODE="worker" ;;
    --all) MODE="all" ;;
    --train) MODE="train" ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg (use --train | --worker | --all)" >&2
      exit 1
      ;;
  esac
done

need_train=0
need_worker=0
case "$MODE" in
  train) need_train=1 ;;
  worker) need_worker=1 ;;
  all) need_train=1; need_worker=1 ;;
esac

echo "==> host check"
uname_s="$(uname -s)"
uname_m="$(uname -m)"
if [[ "$uname_s" != "Darwin" || "$uname_m" != "arm64" ]]; then
  echo "warning: expected Apple Silicon macOS (Darwin/arm64), got ${uname_s}/${uname_m}" >&2
fi

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    echo "uv: $(uv --version)"
    return
  fi
  echo "==> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || {
    echo "uv install failed; install from https://docs.astral.sh/uv/ and re-run" >&2
    exit 1
  }
}

setup_train() {
  echo "==> trainer env (uv sync: TRL SFT/GRPO + LightEval)"
  ensure_uv
  if [[ ! -f .python-version ]]; then
    echo "3.12" > .python-version
  fi
  uv python install "$(cat .python-version)"
  uv sync
  echo
  echo "ready:"
  echo "  bash scripts/train_sft.sh --max-steps 2"
  echo "  ./evals.sh HuggingFaceTB/SmolLM-360M-Instruct --limit 8"
  echo "  bash scripts/eval.sh --suite math --limit 8"
  echo "  bash scripts/train_grpo.sh --max-steps 1 --skip-initial-sync"
}

setup_worker() {
  echo "==> worker env (vllm-metal) — only needed for GRPO rollouts"
  if command -v brew >/dev/null 2>&1; then
    if ! brew tap | grep -q '^vllm-project/vllm-metal$'; then
      brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
    fi
    if ! brew list --formula vllm-project/vllm-metal/vllm-metal >/dev/null 2>&1; then
      brew install vllm-project/vllm-metal/vllm-metal
    else
      echo "vllm-metal already installed via brew"
    fi
  else
    echo "Homebrew not found — using upstream install.sh → ~/.venv-vllm-metal"
    curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
  fi

  mkdir -p "${HOME}/grpo_adapters/default/grpo"
  echo
  echo "worker ready: bash scripts/serve_worker.sh"
}

if (( need_train )); then setup_train; fi
if (( need_worker )); then setup_worker; fi

echo
echo "==> done (${MODE})"
echo "configs: configs/sft.yaml | configs/grpo.yaml | configs/eval.yaml"
