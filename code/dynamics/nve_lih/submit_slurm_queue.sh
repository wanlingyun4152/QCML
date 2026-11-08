#!/bin/bash
set -euo pipefail
PKG=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export QCML_NVE_PACKAGE="$PKG"
export QCML_BASE="${QCML_BASE:-/online1/huangdq_group/liujie/taoyx/trans}"
export QCML_PARENT_MD_SUITE="${QCML_PARENT_MD_SUITE:-$QCML_BASE/revision_runs/qcml_aimd_dt25_sk_prqyi}"
export QCML_OUTPUT_PARENT="${QCML_OUTPUT_PARENT:-$QCML_BASE/revision_runs}"
export QCML_MD_PYTHON="${QCML_MD_PYTHON:-/home/huangdq_group/liujie/.conda/envs/test_2/bin/python}"
[[ -x "$QCML_MD_PYTHON" ]] || { echo "Python not executable: $QCML_MD_PYTHON" >&2; exit 2; }
cd "$PKG"
if [[ -e submitted_jobid.txt ]]; then
  echo 'This package directory already recorded an accepted submission:' >&2
  cat submitted_jobid.txt >&2
  echo 'Do not submit again. Inspect the existing job or use a separate copy only for an explicitly new run.' >&2
  exit 2
fi
mkdir .submit_lock 2>/dev/null || { echo 'Submission already in progress; inspect .submit_lock before retrying.' >&2; exit 2; }
trap 'rmdir .submit_lock 2>/dev/null || true' EXIT
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$QCML_MD_PYTHON" -B -m unittest discover -s tests -v
"$QCML_MD_PYTHON" -B - <<'PY'
import os,sys
from pathlib import Path
from common import verify_package,safe_parent
from staging import check_parent
p=Path.cwd();print(verify_package(p))
parent=Path(os.environ['QCML_PARENT_MD_SUITE']).resolve()
print(check_parent(parent))
safe_parent(os.environ['QCML_OUTPUT_PARENT'],[p,parent])
print('AIMD python =',sys.executable)
print('submission_policy = SLURM_DECIDES')
from common import load
pr=load(p/'protocol.json')
print('MD = corrected force / original prediction, dt=%s, frames=%s, end_time=%s au; no VQE optimization' % (pr['dt_au'],pr['frames'],pr['end_time_au']))
PY
# No squeue call and no fabricated local limit. Slurm alone accepts or rejects.
rc=0
sbatch --parsable --export=ALL --partition="${QCML_PARTITION:-intel}" \
  --time="${QCML_NVE_TIME:-24:00:00}" --mem="${QCML_NVE_MEMORY:-8G}" \
  --chdir="$PKG" "$PKG/run_nve.slurm" > submission_response.txt 2> submission_stderr.txt || rc=$?
cat submission_stderr.txt >&2
if [[ "$rc" -ne 0 ]]; then
  cat submission_response.txt
  echo 'Slurm rejected this attempt; no automatic retry.' >&2
  exit "$rc"
fi
response=$(tr -d '\r\n' < submission_response.txt)
if [[ ! "$response" =~ ^[0-9]+(\;[A-Za-z0-9_.-]+)?$ ]]; then
  printf '%s\n' "$response" > submitted_jobid.txt
  echo 'Submission returned success but an unfamiliar response. Inspect Slurm before any resubmission.' >&2
  exit 2
fi
printf '%s\n' "$response" > submitted_jobid.txt
printf 'Submitted job = %s\n' "$response"
