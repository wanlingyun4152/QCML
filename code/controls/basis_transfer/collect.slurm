#!/usr/bin/env bash
#SBATCH --job-name=qcml-transfer-collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$QCML_TRANSFER_PACKAGE"
"$QCML_PYTHON" transfer.py collect --suite "$QCML_TRANSFER_SUITE"
