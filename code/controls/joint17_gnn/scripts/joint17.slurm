#!/usr/bin/env bash
#SBATCH --job-name=QCML-GNN17
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
set -euo pipefail

qcml_package=$1
qcml_prepared=$2
qcml_results=$3
qcml_python=$4
qcml_task=${SLURM_ARRAY_TASK_ID:?Run through sbatch as an array}
case "$qcml_task" in
  0|1|2) qcml_model=gnn ;;
  3|4|5) qcml_model=transformer ;;
  *) printf 'Unexpected array index %s\n' "$qcml_task" >&2; exit 2 ;;
esac
qcml_seed=$(( qcml_task % 3 ))
exec "$qcml_python" -B -u "$qcml_package/qcml.py" train \
  --model "$qcml_model" --seed "$qcml_seed" --prepared "$qcml_prepared" \
  --threads "${SLURM_CPUS_PER_TASK:-1}" --out "$qcml_results/$qcml_model/seed$qcml_seed"
