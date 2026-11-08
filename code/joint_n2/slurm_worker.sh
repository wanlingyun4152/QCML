#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd -P)"  # Set by --chdir at submission; Slurm may spool the script.
stage="$1"
seed="$2"
mode="$3"
export QCML_PYTHON="$4" OUT_ROOT="$5" QCML_LABELS="$6" QCML_EPOCHS="$7"
export QCML_THREADS="${SLURM_CPUS_PER_TASK:-2}"

if [[ "$mode" == pair ]]; then
    case "${SLURM_ARRAY_TASK_ID:-}" in
        0) mode=legacy ;;
        1) mode=joint ;;
        *) printf 'Expected Slurm array task 0 or 1\n' >&2; exit 2 ;;
    esac
fi
printf 'WORKER node=%s job=%s seed=%s mode=%s labels=%s threads=%s\n' \
    "$(hostname)" "${SLURM_JOB_ID:-unknown}" "$seed" "$mode" "$QCML_LABELS" "$QCML_THREADS"
exec bash "$ROOT/cluster.sh" "$stage" "$seed" "$mode"
