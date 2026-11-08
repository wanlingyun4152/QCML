#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PY="${QCML_PYTHON:-$(command -v python || true)}"
LABELS="${QCML_LABELS:-continuous}"
THREADS="${QCML_THREADS:-2}"
EPOCHS="${QCML_EPOCHS:-3000}"
OUT="$(realpath -m -- "${OUT_ROOT:-$(dirname -- "$ROOT")/revision_runs/qcml_n2_joint_lih_${LABELS}_v1}")"
ACTION="${1:-help}"
SEED="${2:-0}"
MODE="${3:-pair}"

die() { printf '%s\n' "$*" >&2; exit 2; }
usage() {
    cat <<'EOF'
bash cluster.sh preflight
bash cluster.sh submit [seed] [pair|legacy|joint|n2]
bash cluster.sh status [seed]
bash cluster.sh pack [seed]
bash cluster.sh submit-eval [seed] {legacy|joint|n2}
On an allocated compute node:
bash cluster.sh run [seed] {legacy|joint|n2}
bash cluster.sh evaluate [seed] {legacy|joint|n2}
EOF
}

[[ "$ACTION" == help || "$ACTION" == --help ]] && { usage; exit 0; }
[[ "$LABELS" == continuous || "$LABELS" == archived ]] || die "Invalid QCML_LABELS"
[[ "$SEED" =~ ^(0|[1-9][0-9]{0,6})$ ]] || die "Seed must be a nonnegative integer"
[[ "$THREADS" =~ ^[1-9][0-9]*$ && "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || die "Invalid CPU count or epoch count"
[[ "$OUT" != "$ROOT" && "$OUT" != "$ROOT/"* ]] || die "Set OUT_ROOT outside the package directory"

check_python() { [[ "$PY" = /* && -x "$PY" ]] || die "Activate the previous QCML environment and set QCML_PYTHON to its executable Python"; }
check_mode() { [[ "$MODE" == legacy || "$MODE" == joint || "$MODE" == n2 ]] || die "Specify legacy, joint or n2"; }
verify_bundle() {
    (cd "$ROOT" && sha256sum --quiet --strict -c SHA256SUMS) ||
        die "Input checksum failed: use a newly transferred and verified copy of the ZIP"
}
export QCML_THREADS="$THREADS" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
engine() { "$PY" -B "$ROOT/engine/paired.py" "$@" --labels "$LABELS" --threads "$THREADS"; }
preflight() { check_python; verify_bundle; engine preflight --mode joint; }
run_dir() { printf '%s/%s_seed%s\n' "$OUT" "$1" "$SEED"; }
train_complete() {
    local run="$1"
    [[ -s "$run/training.json" && -s "$run/best.pt" && -s "$run/TRAIN_SHA256SUMS" ]] ||
        die "Training is not complete in $run"
    (cd "$run" && sha256sum --quiet --strict -c TRAIN_SHA256SUMS) ||
        die "Trained files failed checksum in $run"
}
test_complete() {
    local run="$1" mode="$2" files
    train_complete "$run"
    files=(training.json history.csv best.pt TRAIN_SHA256SUMS test/metrics.json)
    if [[ "$mode" != n2 ]]; then
        files+=(test/lih_per_geometry.csv test/lih_predictions.npz)
    fi
    if [[ "$mode" != legacy ]]; then
        files+=(test/n2_per_geometry.csv test/n2_predictions.npz)
    fi
    (cd "$run" && sha256sum -- "${files[@]}" > RUN_SHA256SUMS.tmp && mv -- RUN_SHA256SUMS.tmp RUN_SHA256SUMS)
    printf 'RUN_COMPLETE %s\n' "$run"
}
eval_run() {
    local run="$1" staged
    train_complete "$run"
    [[ ! -e "$run/test" ]] || die "Evaluation directory already exists: $run/test"
    staged="$(mktemp -d "$run/evaluation_attempt.XXXXXXXX")"
    engine evaluate --mode "$MODE" --seed "$SEED" --checkpoint "$run/best.pt" --out "$staged/test"
    mv -T -- "$staged/test" "$run/test"
    rmdir -- "$staged"
    test_complete "$run" "$MODE"
}
submit() {
    local stage="$1" slot dir jid log_id='%j'
    local -a jobs array opts
    command -v sbatch >/dev/null || die "sbatch is unavailable here; run this command on the cluster login node"
    if [[ "$MODE" == pair && "$stage" == run ]]; then
        jobs=(legacy joint); array=(--array=0-1); log_id='%A_%a'
    else
        check_mode; jobs=("$MODE"); array=()
    fi
    for slot in "${jobs[@]}"; do
        dir="$(run_dir "$slot")"
        if [[ "$stage" == run ]]; then
            [[ ! -e "$dir" ]] || die "Existing run: $dir. Use a new OUT_ROOT, or submit-eval for a completed train."
        else
            train_complete "$dir"
            [[ ! -e "$dir/test" ]] || die "Existing evaluation: $dir/test"
        fi
    done
    preflight
    mkdir -p -- "$OUT/slurm" "$OUT/submissions"
    local receipt="$OUT/submissions/${stage}_${MODE}_seed${SEED}"
    mkdir -- "$receipt" || die "Submission receipt already exists: $receipt"
    opts=(--parsable --export=ALL --nodes=1 --ntasks=1 --job-name="qcml_lih_${MODE}_s${SEED}"
          --partition="${QCML_PARTITION:-intel}" --cpus-per-task="$THREADS"
          --mem="${QCML_MEM:-4G}" --time="${QCML_TIME:-03:00:00}"
          --chdir="$ROOT" --output="$OUT/slurm/%x_$log_id.out" --error="$OUT/slurm/%x_$log_id.err")
    if ! jid="$(sbatch "${opts[@]}" "${array[@]}" "$ROOT/slurm_worker.sh" "$stage" "$SEED" "$MODE" "$PY" "$OUT" "$LABELS" "$EPOCHS")"; then
        rmdir -- "$receipt"
        die "Slurm did not accept the job; see the scheduler message above"
    fi
    printf '%s\n' "$jid" > "$receipt/jobid"
    printf 'SUBMITTED=%s\nOUT_ROOT=%s\n' "$jid" "$OUT"
}

case "$ACTION" in
    preflight)
        preflight
        ;;
    submit)
        submit run
        ;;
    submit-eval)
        submit evaluate
        ;;
    run|evaluate)
        check_python; check_mode; verify_bundle
        run="$(run_dir "$MODE")"
        if [[ "$ACTION" == run ]]; then
            [[ ! -e "$run" ]] || die "Run already exists: $run"
            engine train --mode "$MODE" --seed "$SEED" --epochs "$EPOCHS" --out "$run"
            (cd "$run" && sha256sum training.json history.csv best.pt > TRAIN_SHA256SUMS.tmp &&
                mv -- TRAIN_SHA256SUMS.tmp TRAIN_SHA256SUMS)
        fi
        eval_run "$run"
        ;;
    status)
        printf 'OUT_ROOT=%s\n' "$OUT"
        shopt -s nullglob
        for file in "$OUT"/submissions/*_seed"$SEED"/jobid; do
            jid="$(<"$file")"
            printf '%s JOB_ID=%s\n' "$(basename -- "$(dirname -- "$file")")" "$jid"
            if command -v squeue >/dev/null; then
                squeue -j "${jid%%;*}" -o "%.18i %.9P %.20j %.8T %.10M %R" || true
            fi
        done
        for mode in legacy joint n2; do
            run="$(run_dir "$mode")"
            if [[ -s "$run/RUN_SHA256SUMS" ]]; then state="TEST_COMPLETE"
            elif [[ -s "$run/TRAIN_SHA256SUMS" ]]; then state="TRAIN_COMPLETE"
            elif [[ -d "$run" ]]; then state="INCOMPLETE_OR_RUNNING"
            else state="NO_RUN_DIRECTORY"
            fi
            printf '%s %s\n' "$mode" "$state"
            if [[ -s "$run/history.csv" ]]; then tail -n 1 -- "$run/history.csv"; fi
        done
        ;;
    pack)
        check_python
        "$PY" -B "$ROOT/engine/collect.py" --root "$OUT" --seed "$SEED" --labels "$LABELS"
        ;;
    *)
        usage
        exit 2
        ;;
esac
