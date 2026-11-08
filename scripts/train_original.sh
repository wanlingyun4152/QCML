#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ $# == 1 ]] || { echo "Usage: bash scripts/train_original.sh NEW_OUTPUT_DIRECTORY" >&2; exit 2; }
PY="${QCML_PYTHON:-python}"
OUT="$(realpath -m -- "$1")"
[[ "$OUT" != "$ROOT" && "$OUT" != "$ROOT/"* ]] || { echo "Use a fresh output directory outside the repository" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "Output already exists; choose a new directory" >&2; exit 2; }
mkdir -p -- "$(dirname -- "$OUT")"
mkdir -- "$OUT"
mkdir -- "$OUT/pre"
export QCML_DATA_DIR="${QCML_DATA_DIR:-$ROOT/data/legacy_raw17}"
cd -- "$OUT"
exec "$PY" "$ROOT/code/original_transformer/2h2l/nn.py"
