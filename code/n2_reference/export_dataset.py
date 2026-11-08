"""Export one audited UCCSD label per geometry, with a frozen geometry split."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "reference"))
from qsector.algebra import pool

FEATURES = ["R_angstrom", "n_spatial", "n_electrons", "n_parameters",
            "HOMO_canonical_Eh", "LUMO_canonical_Eh"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")


def export(root, out):
    root, out = root.resolve(), out.resolve()
    cfg = read(root / "config.json")
    if (cfg["basis"], cfg["ncore"], cfg["ncas"], cfg["nelecas"],
        cfg["ansatz"], cfg["layers"]) != ("sto-3g", 2, 8, [5, 5], "uccsd", 1):
        raise ValueError("Only this frozen-core STO-3G single-layer UCCSD parameter space is supported")
    if cfg["bond_lengths_angstrom"] != [i/100 for i in range(90, 351, 5)]:
        raise ValueError("This split requires the exact 53-point grid; do not silently change it")
    ops, labels = pool(8, 5, "uccsd")
    if len(ops) != 135:
        raise ValueError("Operator count mismatch")
    param_map = {"family": "uccsd", "layers": 1,
                 "coefficient_normalization": "unit L2 norm of normal-ordered fermion coefficients",
                 "base_generators": [{"index": i, "label": label, "terms": [
                     {"word": [list(q) for q in word], "coefficient": float(c)}
                     for word, c in sorted(op.terms.items())]}
                     for i, (op, label) in enumerate(zip(ops, labels))]}
    rows, features, theta = [], [], []
    for i, r in enumerate(cfg["bond_lengths_angstrom"]):
        ref_dir = root / "references" / f"r{i:02d}"
        vqe_dir = root / "vqe" / f"r{i:02d}_seed0"
        ref, fit = read(ref_dir/"result.json"), read(vqe_dir/"result.json")
        if not (all(ref["checks"].values()) and fit["converged"] and fit["numerical_checks_passed"]
                and fit["index"] == ref["index"] == i and fit["seed"] == 0
                and fit["family"] == "uccsd" and fit["layers"] == 1
                and fit["n_parameters"] == 135 and fit["R_angstrom"] == ref["R_angstrom"] == r
                and not fit["CASCI_used_for_initialization"]
                and not fit["cross_geometry_warm_start"]
                and fit["gradient_inf"] < cfg["thresholds"]["vqe_gradient_inf"]):
            raise ValueError(f"Unqualified label at index {i}, R={r}: inspect the raw run")
        if (sha(ref_dir/"reference.npz") != ref["reference_sha256"] or
                fit["reference_sha256"] != ref["reference_sha256"] or
                sha(vqe_dir/"optimized.npz") != fit["optimized_sha256"]):
            raise ValueError(f"Gauge/reference/label checksum mismatch at index {i}")
        with np.load(vqe_dir/"optimized.npz", allow_pickle=False) as z:
            y = z["theta"]
        if y.shape != (135,) or not np.isfinite(y).all():
            raise ValueError(f"Invalid parameter vector at index {i}")
        features.append([r, 8, 10, 135, ref["HOMO_Eh"], ref["LUMO_Eh"]])
        theta.append(y)
        split = {1: "val", 3: "test"}.get(i % 6, "train")
        rows.append({"index": i, "R_angstrom": r, "split": split, "vqe_seed": 0,
                     "E_VQE_Eh": fit["E_VQE_Eh"], "E_CASCI_Eh": ref["E_CASCI_Eh"],
                     "VQE_minus_CASCI_mHa": 1000*(fit["E_VQE_Eh"]-ref["E_CASCI_Eh"]),
                     "VQE_fidelity_to_CASCI": fit["fidelity_to_CASCI"],
                     "gradient_inf": fit["gradient_inf"],
                     "orbital_overlap_min": ref["min_matched_orbital_abs_overlap"],
                     "reference_sha256": ref["reference_sha256"],
                     "optimized_sha256": fit["optimized_sha256"]})
    x, y = np.asarray(features, dtype=np.float64), np.asarray(theta, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite features")
    out.mkdir(parents=True, exist_ok=False)
    write(out/"parameter_map.json", param_map)
    write(out/"geometry_manifest.json", rows)
    with (out/"geometry_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for split in ("train", "val", "test"):
        idx = np.array([r["index"] for r in rows if r["split"] == split])
        np.savez_compressed(out/f"{split}.npz", index=idx, x=x[idx], theta=y[idx])
        counts[split] = len(idx)
    jumps = np.sqrt(np.mean(np.diff(y, axis=0)**2, axis=1))
    k = int(jumps.argmax())
    meta = {"schema": "qcml_n2_local_training_v1", "feature_names": FEATURES,
            "counts": counts, "n_parameters": 135, "label": "converged UCCSD-VQE theta",
            "source_config_sha256": sha(root/"config.json"),
            "input_scaling": {"offset": [1.8, 0, 0, 0, 0, 0],
                              "scale": [1, 20, 4, 640, 1, 1]},
            "split_rule": "index mod 6 == 1 validation, == 3 test, otherwise training",
            "audit": {"max_gradient_inf": max(r["gradient_inf"] for r in rows),
                      "min_matched_orbital_overlap": min(r["orbital_overlap_min"] for r in rows),
                      "max_adjacent_parameter_rmse": float(jumps[k]),
                      "largest_adjacent_change_R_angstrom": [rows[k]["R_angstrom"], rows[k+1]["R_angstrom"]],
                      "max_UCCSD_minus_CASCI_mHa": max(r["VQE_minus_CASCI_mHa"] for r in rows),
                      "noon_and_CASCI_not_model_features": True},
            "sha256": {p.name: sha(p) for p in sorted(out.iterdir())}}
    write(out/"manifest.json", meta)
    print(json.dumps({"counts": counts, "audit": meta["audit"]}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=HERE/"runs/uccsd_dense")
    p.add_argument("--out", type=Path, default=HERE/"dataset/uccsd_53")
    a = p.parse_args()
    export(a.root, a.out)

