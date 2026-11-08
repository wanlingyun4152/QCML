#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
jobs="${N2_JOBS:-4}"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "N2_JOBS must be positive" >&2; exit 2; }
root="${1:-runs/uccsd_dense_local}"
config=configs/uccsd_dense.json
export QCML_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON_BIN" reference/n2_benchmark.py references --config "$config" --root "$root"
seq 0 52 | xargs -P "$jobs" -I '{}' "$PYTHON_BIN" reference/n2_benchmark.py vqe --config "$config" --root "$root" --index '{}' --seed 0
echo "Dense UCCSD run: $root"

