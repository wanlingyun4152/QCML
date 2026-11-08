"""Verify and package a completed pair, including directly comparable test metrics."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TEST_FILES = ("test/metrics.json", "test/lih_per_geometry.csv", "test/lih_predictions.npz")
N2_FILES = ("test/n2_per_geometry.csv", "test/n2_predictions.npz")
FIT_FILES = ("training.json", "history.csv", "best.pt", "TRAIN_SHA256SUMS")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(run, mode, seed, labels):
    expected = set(FIT_FILES + ("test/metrics.json",) +
                   (TEST_FILES[1:] if mode != "n2" else ()) +
                   (N2_FILES if mode != "legacy" else ()))
    marker = run/"RUN_SHA256SUMS"
    if not marker.is_file():
        raise ValueError(f"Missing completion record: {marker}")
    entries = {}
    for line in marker.read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
        if not m or m[2] in entries or m[2] not in expected:
            raise ValueError(f"Unexpected checksum entry in {marker}")
        entries[m[2]] = m[1]
    if set(entries) != expected:
        raise ValueError(f"Missing result checksum in {marker}")
    for name, want in entries.items():
        if digest((run/name).read_bytes()) != want:
            raise ValueError(f"Output checksum mismatch: {run/name}")
    fit, test = read_json(run/"training.json"), read_json(run/"test/metrics.json")
    identity = fit.get("legacy_identity")
    if mode != "n2" and (not identity or identity["lih_label_variant"] != labels):
        raise ValueError(f"Unexpected legacy training labels: {run}")
    if (fit["mode"] != mode or test["mode"] != mode or fit["seed"] != seed or test["seed"] != seed or
            test["training_label_variant"] != labels or fit["test_used_for_fitting"] or test["split"] != "test" or
            test["epochs"] != fit["epochs"] or
            fit["best_epoch"] != test["best_epoch"] or entries["best.pt"] != test["checkpoint_sha256"] or
            test.get("legacy_identity") != identity or test["post_prediction_vqe_steps"] != 0):
        raise ValueError(f"Training/evaluation mismatch: {run}")
    with (run/"history.csv").open(newline="", encoding="utf-8") as f:
        history = list(csv.DictReader(f))
    if ([int(h["epoch"]) for h in history] != list(range(1,fit["epochs"]+1)) or
            not all(math.isfinite(float(h["selection_mse"])) for h in history) or
            int(min(history, key=lambda h: float(h["selection_mse"]))["epoch"]) != fit["best_epoch"]):
        raise ValueError(f"History does not reproduce the selected checkpoint: {run}")
    return fit, test, sorted(expected | {"RUN_SHA256SUMS"})


def comparison_row(m, group):
    lih = m.get("LiH_vs_continuous_VQE", {})
    new = m.get("legacy_parameter_test", {}).get("all", {})
    old = m.get("legacy_parameter_test_archived_reference", {}).get("all", {})
    other = m.get("legacy_parameter_test_excluding_relabelled_task", {})
    n2 = m.get("N2_vs_VQE", {}).get(m["mode"], {})
    return dict(group=group, mode=m["mode"], seed=m["seed"],
                epochs=m["epochs"], best_epoch=m["best_epoch"], label_version=m["training_label_variant"],
                legacy_theta_RMSE_common=new.get("theta_RMSE_micro"),
                legacy_theta_RMSE_archived_targets=old.get("theta_RMSE_micro"),
                unchanged_16_tasks_theta_RMSE=other.get("theta_RMSE_micro"),
                LiH_theta_RMSE=lih.get("theta_RMSE_continuous"),
                LiH_energy_RMSE_mHa=lih.get("energy_RMSE_mHa"),
                LiH_focus_energy_RMSE_mHa=lih.get("focus_energy_RMSE_mHa"),
                LiH_min_fidelity=lih.get("fidelity_to_VQE_min"),
                N2_theta_RMSE=n2.get("theta_RMSE"),
                N2_energy_RMSE_mHa=n2.get("energy_RMSE_mHa"),
                N2_min_fidelity=n2.get("fidelity_to_VQE_min"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--labels", choices=("continuous", "archived"), default="continuous")
    args = p.parse_args()
    root = args.root.resolve()
    modes = ["legacy", "joint"]
    if (root/f"n2_seed{args.seed}"/"RUN_SHA256SUMS").is_file():
        modes.append("n2")
    pairs = {}
    for mode in modes:
        run = root/f"{mode}_seed{args.seed}"
        pairs[mode] = verify(run, mode, args.seed, args.labels)
    a, x, _ = pairs["legacy"]
    b, y, _ = pairs["joint"]
    match = ("epochs", "batch_size", "device", "torch", "legacy_identity")
    eval_match = ("evaluation_threads", "evaluation_torch", "lih_reference_manifest_sha256",
                  "n2_reference_manifest_sha256")
    if (any(a[k] != b[k] for k in match) or any(x[k] != y[k] for k in eval_match) or
            a["train_counts"]["old"] != b["train_counts"]["old"] or
            a["val_counts"]["old"] != b["val_counts"]["old"]):
        raise ValueError("legacy/joint runs are not a matched pair")
    if "n2" in pairs:
        c, z, _ = pairs["n2"]
        if (any(c[k] != b[k] for k in ("epochs","batch_size","device","torch")) or
                any(z[k] != y[k] for k in eval_match) or
                any(c[k]["n2"] != b[k]["n2"] for k in ("train_counts","val_counts"))):
            raise ValueError("N2-only run does not match the joint settings")

    archive = {}
    def add(path, relative):
        archive[relative] = path.read_bytes()
    rows = [comparison_row(pairs[mode][1], "this_run") for mode in modes]
    for mode in modes:
        run = root/f"{mode}_seed{args.seed}"
        for f in pairs[mode][2]:
            add(run/f, run.name+"/"+f)
    if args.seed == 0:
        for mode in ("legacy", "joint", "n2"):
            path = ROOT/"baseline"/"seed0"/mode/"metrics.json"
            meta = read_json(path)
            if any(meta[k] != x[k] for k in ("lih_reference_manifest_sha256", "n2_reference_manifest_sha256")):
                raise ValueError("Historical baselines use different test references")
            rows.append(comparison_row(meta, "historical_seed0"))
            add(path, "baseline_seed0/"+mode+"/metrics.json")
        add(ROOT/"baseline"/"seed0"/"provenance.json", "baseline_seed0/provenance.json")
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    archive["comparison.csv"] = buf.getvalue().encode()
    archive["SHA256SUMS"] = "".join(f"{digest(archive[k])}  {k}\n" for k in sorted(archive)).encode()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = root/f"qcml_lih_{args.labels}_seed{args.seed}_{stamp}.zip"
    with zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, data in sorted(archive.items()):
            z.writestr(name, data)
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:
            raise RuntimeError(f"Unreadable result archive: {out}")
    print(f"RESULT_ZIP={out}")
    print(f"SHA256={digest(out.read_bytes())}")


if __name__ == "__main__":
    main()
