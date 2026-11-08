#!/bin/bash
#SBATCH --job-name=qcml-lih
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
set -euo pipefail
: "${QCML_PYTHON:?}" "${QCML_PARALLEL_DIR:?}" "${QCML_LIH_SUITE:?}"
cd "$QCML_PARALLEL_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mode="${1:?smoke, task or collect}"
case "$mode" in
 smoke) exec "$QCML_PYTHON" parallel.py smoke --suite "$QCML_LIH_SUITE" ;;
 task) exec "$QCML_PYTHON" parallel.py task --suite "$QCML_LIH_SUITE" --index "${SLURM_ARRAY_TASK_ID:?}" ;;
 collect) exec "$QCML_PYTHON" parallel.py collect --suite "$QCML_LIH_SUITE" ;;
 *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
