#!/usr/bin/env bash
set -euo pipefail
PKG=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-pilot}
case "$MODE" in smoke|pilot|formal) ;; *) echo 'mode must be smoke, pilot or formal' >&2; exit 2;; esac
PY=${QCML_PYTHON:-python3}
DATA=${QCML_DATA_ROOT:-/online1/huangdq_group/liujie/taoyx/trans/data}
PARENT=${QCML_OUTPUT_PARENT:-/online1/huangdq_group/liujie/taoyx/trans/revision_runs}
THREADS=${QCML_THREADS:-${SLURM_CPUS_PER_TASK:-1}}
DEVICE=${QCML_DEVICE:-cpu}
SEEDS=${QCML_SEEDS:-0}
LOSSES=${QCML_LOSSES:-weighted}
EPOCHS=${QCML_EPOCHS:-3000}
if [[ "$MODE" == smoke ]]; then EPOCHS=3; SEEDS=0; LOSSES=weighted; fi
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
cd "$PKG"
"$PY" qcml.py check-env
# Reject output under original data/train before creating any directory there.
SUITE=$("$PY" - "$DATA" "$PARENT" "$MODE" <<'PY'
import os,sys,tempfile
from pathlib import Path
data,parent,mode=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve(),sys.argv[3]
if not data.is_dir(): raise SystemExit('Data root does not exist: '+str(data))
for protected in [data,data.parent/'train']:
    if parent==protected or protected in parent.parents:
        raise SystemExit('Output parent must be outside original data/train: '+str(parent))
parent.mkdir(parents=True,exist_ok=True)
print(tempfile.mkdtemp(prefix='qcml_independent_'+mode+'_',dir=str(parent)))
PY
)
echo "suite_dir = $SUITE"
exec > >(tee -a "$SUITE/suite.log") 2>&1
finish() {
  rc=$?
  trap - EXIT
  set +e
  "$PY" - "$SUITE" "$rc" "$MODE" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);rc=int(sys.argv[2])
(p/'suite_status.json').write_text(json.dumps({'exit_code':rc,
 'status':'COMPLETED' if rc==0 else 'FAILED','purpose':sys.argv[3],
 'test_evaluated':False,'note':'No automatic final-test evaluation or quantum calculation.'},indent=2)+'\n')
PY
  "$PY" qcml.py collect --root "$SUITE"
  collect_rc=$?
  if [[ $rc -eq 0 && $collect_rc -ne 0 ]]; then rc=$collect_rc; fi
  echo "suite_exit_code = $rc"
  echo "suite_dir = $SUITE"
  exit "$rc"
}
trap finish EXIT
"$PY" qcml.py test --report "$SUITE/TEST_RESULTS.json" > "$SUITE/test_stdout.txt" 2>&1
"$PY" qcml.py prepare --data-root "$DATA" --out "$SUITE/prepared" --seed 42
for seed in $SEEDS; do
  for loss in $LOSSES; do
    for model in transformer mlp; do
      RUN="$SUITE/${model}_seed${seed}_${loss}"
      "$PY" qcml.py train --prepared "$SUITE/prepared" --config "$PKG/configs/${model}.json" \
        --out "$RUN" --seed "$seed" --epochs "$EPOCHS" --loss "$loss" \
        --purpose "$MODE" --device "$DEVICE" --threads "$THREADS"
      "$PY" qcml.py evaluate --run "$RUN" --prepared "$SUITE/prepared" --split val \
        --device "$DEVICE" --threads "$THREADS"
    done
  done
done
echo 'SUITE TRAINING / VALIDATION COMPLETE; final test remains locked.'
