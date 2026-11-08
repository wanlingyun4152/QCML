#!/usr/bin/env bash
#SBATCH --job-name=qcml-basis-transfer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$QCML_TRANSFER_PACKAGE"
"$QCML_PYTHON" transfer.py worker --suite "$QCML_TRANSFER_SUITE" \
 --index "$SLURM_ARRAY_TASK_ID" --count "$QCML_TRANSFER_WORKERS"
