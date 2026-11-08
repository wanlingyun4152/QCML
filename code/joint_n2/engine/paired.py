"""Train or evaluate the original QCML Transformer with the verified LiH overlay."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from joint import MODES, SharedTransformer, fit, json_out, old_data, old_metrics, predict
from legacy_inputs import sha
from lih_data import OVERLAY_MANIFEST_SHA, TASK, Overlay, evaluate_lih
from vqe_only import csv_out, evaluate_n2, n2_data


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT/"inputs"/"legacy"
LIH = ROOT/"inputs"/"lih"
N2 = ROOT/"inputs"/"n2"
N2_MANIFEST = N2/"dataset"/"uccsd_53"/"manifest.json"


def load(action, mode, labels):
    splits = ("test",) if action == "evaluate" else ("train", "val")
    domains = MODES[mode]
    datasets, archived, corrected, refs = {}, {}, {}, {}
    overlay, identity = None, None
    if "old" in domains:
        archived, base_id = old_data(LEGACY, splits)
        overlay = Overlay(LIH, LEGACY, base_id)
        if labels == "continuous" or action == "evaluate":
            for s in splits:
                corrected[s], refs[s] = overlay.apply(s, archived[s])
        datasets["old"] = corrected if labels == "continuous" else archived
        identity = dict(base_id, lih_label_variant=labels,
                        lih_overlay_manifest_sha256=OVERLAY_MANIFEST_SHA if labels == "continuous" else None)
    if "n2" in domains:
        datasets["n2"] = n2_data(N2, splits)
    return datasets, archived, corrected, refs, overlay, identity


def without_task(row, pred):
    keep = row["task"] != TASK
    return old_metrics(pred[keep], {k: v[keep] for k, v in row.items()})["all"]


def load_checkpoint(path, mode, seed, labels, identity):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["mode"] != mode or int(ck["seed"]) != seed:
        raise ValueError("Checkpoint mode/seed differs from the requested run")
    if identity is not None:
        original = ck.get("legacy_identity", {})
        if (original.get("files_sha256") != identity["files_sha256"] or
                original.get("source_fingerprint") != identity["source_fingerprint"] or
                original.get("lih_label_variant", "archived") != labels or
                original.get("lih_overlay_manifest_sha256") != identity["lih_overlay_manifest_sha256"]):
            raise ValueError("Checkpoint and requested training label version do not match")
    return ck


def evaluate(mode, seed, labels, checkpoint, out, device, datasets, archived, corrected, refs, overlay, identity):
    ck = load_checkpoint(checkpoint, mode, seed, labels, identity)
    model = SharedTransformer().to(device)
    model.load_state_dict(ck["state"], strict=True)
    model.eval()
    summary = dict(mode=mode, seed=seed, training_label_variant=labels, split="test",
                   epochs=int(ck["epochs"]), best_epoch=int(ck["epoch"]),
                   checkpoint_sha256=sha(checkpoint), trained_device=ck["device"], trained_torch=ck["torch"],
                   evaluation_torch=str(torch.__version__), evaluation_threads=torch.get_num_threads(),
                   lih_reference_manifest_sha256=OVERLAY_MANIFEST_SHA,
                   n2_reference_manifest_sha256=sha(N2_MANIFEST),
                   direct_circuit_evaluation=True, post_prediction_vqe_steps=0)
    extra = {}
    if "old" in datasets:
        p = predict(model, "old", archived["test"], device)
        summary["legacy_identity"] = identity
        summary["legacy_parameter_test"] = old_metrics(p, corrected["test"])
        summary["legacy_parameter_test_archived_reference"] = old_metrics(p, archived["test"])
        summary["legacy_parameter_test_excluding_relabelled_task"] = without_task(archived["test"], p)
        lih, lih_rows, lih_pred = evaluate_lih(N2, overlay, refs["test"], p)
        lih["focus_fraction_of_legacy_theta_SSE_continuous"] = (
            lih["focus_theta_SSE_continuous"] / (
                summary["legacy_parameter_test"]["all"]["theta_RMSE_micro"]**2 *
                summary["legacy_parameter_test"]["all"]["valid_labels"]))
        lih["focus_fraction_of_legacy_theta_SSE_archived"] = (
            lih["focus_theta_SSE_archived"] / (
                summary["legacy_parameter_test_archived_reference"]["all"]["theta_RMSE_micro"]**2 *
                summary["legacy_parameter_test_archived_reference"]["all"]["valid_labels"]))
        summary["LiH_vs_continuous_VQE"] = lih
        extra["lih"] = (lih_rows, lih_pred)
    if "n2" in datasets:
        n2 = datasets["n2"]["test"]
        p = predict(model, "n2", n2, device)
        summary["N2_vs_VQE"], n2_rows = evaluate_n2(N2, n2, {mode: p})
        extra["n2"] = (n2_rows, p)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    if "lih" in extra:
        rows, pred = extra["lih"]
        csv_out(out/"lih_per_geometry.csv", rows)
        np.savez_compressed(out/"lih_predictions.npz", index=refs["test"]["global_rows"],
                            R_A=refs["test"]["R_A"], theta=pred)
    if "n2" in extra:
        rows, pred = extra["n2"]
        csv_out(out/"n2_per_geometry.csv", rows)
        np.savez_compressed(out/"n2_predictions.npz", index=datasets["n2"]["test"]["index"], theta=pred)
    json_out(out/"metrics.json", summary)
    key = {"mode": mode, "seed": seed, "training_labels": labels}
    if "LiH_vs_continuous_VQE" in summary:
        key["LiH_energy_RMSE_mHa"] = summary["LiH_vs_continuous_VQE"]["energy_RMSE_mHa"]
        key["LiH_theta_RMSE"] = summary["LiH_vs_continuous_VQE"]["theta_RMSE_continuous"]
    if "N2_vs_VQE" in summary:
        key["N2_energy_RMSE_mHa"] = summary["N2_vs_VQE"][mode]["energy_RMSE_mHa"]
    print("TEST_COMPLETE", key, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("preflight", "train", "evaluate"))
    p.add_argument("--mode", choices=MODES, default="joint")
    p.add_argument("--labels", choices=("continuous", "archived"), default="continuous")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", type=Path)
    p.add_argument("--checkpoint", type=Path)
    args = p.parse_args()
    if args.seed < 0 or args.epochs < 1 or args.threads < 1:
        p.error("seed >= 0; epochs and threads >= 1")
    if args.action != "preflight" and args.out is None:
        p.error("--out is required")
    if args.action == "evaluate" and args.checkpoint is None:
        p.error("--checkpoint is required for evaluation")
    if args.out is not None:
        inputs = (ROOT/"inputs").resolve()
        dest = args.out.resolve()
        if inputs == dest or inputs in dest.parents or dest.exists():
            p.error("Use a new output directory outside inputs/")
    torch.set_num_threads(args.threads)
    datasets, archived, corrected, refs, overlay, identity = load(args.action, args.mode, args.labels)
    if args.action == "preflight":
        import pyscf
        import scipy
        sizes = {d: {s: len(a["index"]) for s, a in sp.items()} for d, sp in datasets.items()}
        print("PREFLIGHT_OK", sizes, "LiH_overlay", {k:len(v["global_rows"]) for k,v in refs.items()},
              "torch", torch.__version__, "pyscf", pyscf.__version__, "scipy", scipy.__version__, flush=True)
    elif args.action == "train":
        fit(datasets, args.mode, args.seed, args.epochs, "cpu", args.out, legacy_identity=identity)
    else:
        evaluate(args.mode, args.seed, args.labels, args.checkpoint, args.out, "cpu",
                 datasets, archived, corrected, refs, overlay, identity)


if __name__ == "__main__":
    main()
