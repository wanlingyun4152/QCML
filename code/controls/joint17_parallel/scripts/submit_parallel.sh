#!/bin/bash
# NEW LiH-only three-architecture experiment. Does not resubmit the old 3 MLP tasks.
set -euo pipefail
export QCML_PARALLEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${QCML_PYTHON:?Set the previously verified Python interpreter}"
: "${QCML_DATA_ROOT:?}" "${QCML_PARENT_PREPARED:?}" "${QCML_OUTPUT_PARENT:?}"
command -v sbatch >/dev/null || { echo "sbatch not found" >&2; exit 2; }
MAX="${QCML_MAX_CONCURRENT:-6}"
[[ "$MAX" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid concurrency' >&2; exit 2; }
cd "$QCML_PARALLEL_DIR"
"$QCML_PYTHON" qcml.py check-env
export QCML_LIH_SUITE
QCML_LIH_SUITE="$("$QCML_PYTHON" parallel.py init --data-root "$QCML_DATA_ROOT" \
 --parent-prepared "$QCML_PARENT_PREPARED" --out-parent "$QCML_OUTPUT_PARENT")"
echo "suite_dir = $QCML_LIH_SUITE"
ARGS=(--parsable --export=ALL --partition="${QCML_PARTITION:-intel}")
[[ -z "${QCML_ACCOUNT:-}" ]] || ARGS+=(--account="$QCML_ACCOUNT")
SMOKE=$(sbatch "${ARGS[@]}" --output="$QCML_LIH_SUITE/logs/preflight-%j.log" \
 scripts/worker.slurm smoke)
SMOKE=${SMOKE%%;*}
ARRAY=$(sbatch "${ARGS[@]}" --dependency="afterok:$SMOKE" --kill-on-invalid-dep=yes \
 --array="0-17%$MAX" --output="$QCML_LIH_SUITE/logs/array-%A_%a.log" scripts/worker.slurm task)
ARRAY=${ARRAY%%;*}
COLLECT=$(sbatch "${ARGS[@]}" --dependency="afterany:$ARRAY" \
 --output="$QCML_LIH_SUITE/logs/collect-%j.log" scripts/worker.slurm collect)
COLLECT=${COLLECT%%;*}
"$QCML_PYTHON" - "$QCML_LIH_SUITE" "$SMOKE" "$ARRAY" "$COLLECT" <<'PY'
import json,sys,os
p,s,a,c=sys.argv[1:]
with open(os.path.join(p,'submission.json'),'x') as f:
 json.dump({'preflight_job':s,'array_job':a,'collect_job':c,'array':'0-17',
            'test_evaluation_planned':False},f,indent=2)
PY
echo "preflight_job = $SMOKE"
echo "training_array = $ARRAY"
echo "collect_job = $COLLECT"
echo 'Preflight gates all 18 runs. They do not depend on the older MLP matrix job.'
