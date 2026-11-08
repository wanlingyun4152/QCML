#!/usr/bin/env bash
set -euo pipefail
PKG=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${QCML_PYTHON:-python3}
BASE=${QCML_BASE:-/online1/huangdq_group/liujie/taoyx/trans}
DATA=${QCML_TRANSFER_DATASET:-$BASE/revision_runs/qcml_regauge_zhjp6ejq/release/dataset_stage4}
OUT=${QCML_OUTPUT_PARENT:-$BASE/revision_runs}
cd "$PKG"
"$PY" -m unittest discover -s tests -v
"$PY" transfer.py check-env
"$PY" transfer.py submit --dataset "$DATA" --out-parent "$OUT" \
 --partition "${QCML_PARTITION:-intel}" --limit "${QCML_SUBMIT_LIMIT:-40}" \
 --reserve "${QCML_JOB_RESERVE:-2}" --workers "${QCML_TRANSFER_WORKERS:-6}"
