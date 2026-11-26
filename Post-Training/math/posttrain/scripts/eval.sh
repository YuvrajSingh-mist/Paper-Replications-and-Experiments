#!/usr/bin/env bash
# Thin wrapper: math suite by default.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uv run math-eval --config "${CONFIG:-configs/eval.yaml}" --suite "${SUITE:-math}" "$@"
