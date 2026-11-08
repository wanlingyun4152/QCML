#!/usr/bin/env bash
#SBATCH --job-name=qcml_rev_pair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=qcml-revision-%j.log
#SBATCH --error=qcml-revision-%j.err
set -euo pipefail
# Submit from the extracted package root. No partition/account/conda environment is guessed.
PKG=${QCML_PACKAGE_DIR:-${SLURM_SUBMIT_DIR:?Submit from the package root}}
[[ -f "$PKG/qcml.py" ]] || { echo "Bad QCML_PACKAGE_DIR: $PKG" >&2; exit 2; }
# This job uses CPU by default. Request GPUs with your cluster's own flags before QCML_DEVICE=cuda.
exec bash "$PKG/scripts/run_pair.sh" "${QCML_PURPOSE:-pilot}"
