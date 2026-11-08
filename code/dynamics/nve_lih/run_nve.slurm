#!/bin/bash
#SBATCH -J qcml-nve-corrected
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@120
#SBATCH -o nve-corrected-%j.log
#SBATCH -e nve-corrected-%j.err
set -euo pipefail
PKG=${QCML_NVE_PACKAGE:-${SLURM_SUBMIT_DIR:?}}
PY=${QCML_MD_PYTHON:?}
BASE=${QCML_BASE:-/online1/huangdq_group/liujie/taoyx/trans}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
unset PYTHONPATH || true
cd "$PKG"
"$PY" -B -c 'from common import verify_package; from pathlib import Path; print(verify_package(Path.cwd()))'
runner=''
stop_run() {
    trap '' USR1 TERM INT
    if [[ -n "$runner" ]]; then
        kill -TERM "$runner" 2>/dev/null || true
        wait "$runner" || true
    fi
    exit 124
}
trap stop_run USR1 TERM INT
"$PY" -B "$PKG/run_nve.py" run \
    --parent "${QCML_PARENT_MD_SUITE:-$BASE/revision_runs/qcml_aimd_dt25_sk_prqyi}" \
    --out-parent "${QCML_OUTPUT_PARENT:-$BASE/revision_runs}" --python "$PY" &
runner=$!
rc=0
wait "$runner" || rc=$?
exit "$rc"
