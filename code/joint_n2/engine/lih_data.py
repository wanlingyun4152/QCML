"""Verified overlay and VQE evaluation for just the 2-UpCCGSD/LiH task."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

from legacy_inputs import sha


TASK = "2kupccgsd/LiH"
NAMES = [f"VQE_param{i}" for i in range(1, 61)]
COUNTS = {"train": 203, "val": 44, "test": 44}
OVERLAY_MANIFEST_SHA = "a1bca9b4228863024f18a29eee82a913c254c3a86d386364eae1a9f4fb5e8cf4"
FOCUS_R = {105, 106, 107}


class Overlay:
    def __init__(self, root, legacy, identity):
        self.root = Path(root)
        if sha(self.root/"manifest.json") != OVERLAY_MANIFEST_SHA:
            raise ValueError("LiH overlay manifest checksum mismatch")
        self.m = json.loads((self.root/"manifest.json").read_text(encoding="utf-8"))
        if (self.m["schema"] != "qcml_lih_continuous_overlay_v1" or
                self.m["task"] != TASK or self.m["counts"] != COUNTS or
                self.m["parameter_names"] != NAMES or
                self.m["original_prepared_fingerprint"] != identity["source_fingerprint"]):
            raise ValueError("LiH overlay incompatible with the verified original dataset")
        keys = json.loads((legacy/"schema.json").read_text(encoding="utf-8"))["parameter_columns"]
        self.cols = np.array([keys.index(k) for k in NAMES], dtype=np.int64)
        if len(np.unique(self.cols)) != 60:
            raise ValueError("Duplicate mapped LiH parameter column")
        with (legacy/"samples.tsv").open(newline="", encoding="utf-8") as f:
            self.samples = {int(r["global_row"]): r for r in csv.DictReader(f, delimiter="\t")
                            if r["task"] == TASK}

    def file(self, name):
        p = self.root/name
        if sha(p) != self.m["files_sha256"][name]:
            raise ValueError(f"LiH overlay checksum mismatch: {name}")
        return p

    def apply(self, split, original):
        # Only read a split explicitly requested by the caller.
        with self.file(f"{split}.tsv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        global_rows = np.array([int(r["global_row"]) for r in rows], dtype=np.int64)
        task_idx = {int(i): k for k, i in enumerate(original["index"])
                    if original["task"][k] == TASK}
        if (len(rows) != COUNTS[split] or len(task_idx) != len(rows) or
                len(set(global_rows.tolist())) != len(rows) or set(task_idx) != set(global_rows)):
            raise ValueError(f"LiH {split} overlay row identity mismatch")
        theta, energy, distance = [], [], []
        for r, i in zip(rows, global_rows):
            src = self.samples[int(i)]
            R = float(r["R_A"])
            if (src["sample_id"] != r["sample_id"] or r["task"] != TASK or
                    r["split"] != split or src["split"] != split or
                    abs(float(src["bondlengths"]) - R) > 1e-10):
                raise ValueError(f"Wrong sample/geometry for LiH global row {i}")
            theta.append([float(r[k]) for k in NAMES])
            energy.append(float(r["VQE_energy_Eh"]))
            distance.append(R)
        theta, energy, distance = map(np.asarray, (theta, energy, distance))
        if any(not np.isfinite(x).all() for x in (theta, energy, distance)):
            raise ValueError("Nonfinite LiH overlay")
        ij = np.array([task_idx[int(i)] for i in global_rows], dtype=np.int64)
        mask = original["mask"]
        if not (mask[ij].sum(axis=1) == 60).all() or not mask[np.ix_(ij, self.cols)].all():
            raise ValueError("LiH task mask differs from the 60 required circuit angles")
        y = original["y"].copy()
        y[np.ix_(ij, self.cols)] = theta
        new = dict(original, y=y)
        ref = dict(global_rows=global_rows, indices=ij, R_A=distance, theta=theta,
                   energy=energy, archived_theta=original["y"][np.ix_(ij, self.cols)])
        return new, ref


def evaluate_lih(n2_root, overlay, ref, legacy_predictions):
    # Use the same determinant-sector engine and normalized generators as the
    # original-circuit replay. Evaluate predictions directly with no refinement.
    sys.path.insert(0, str(n2_root/"reference"))
    from qsector.algebra import pool
    from qsector.ansatz import Ansatz
    from qsector.sector import CompiledPool, PySCFHamiltonian, Sector
    ops, labels = pool(6, 2, "kupccgsd")
    if len(ops) != 30 or len(labels) != 30:
        raise ValueError("Unexpected LiH generator count")
    compiled = CompiledPool(Sector(6, 2, 2), ops)
    pred = legacy_predictions[np.ix_(ref["indices"], overlay.cols)]
    err_theta = pred - ref["theta"]
    err_old_theta = pred - ref["archived_theta"]
    per_geometry = []
    replay_errors = []
    with np.load(overlay.file("test_integrals.npz"), allow_pickle=False) as z:
        n = len(pred)
        if (not np.array_equal(z["global_row"], ref["global_rows"]) or
                not np.array_equal(z["R_A"], ref["R_A"]) or
                z["h1"].shape != (n, 6, 6) or z["eri"].shape != (n, 6, 6, 6, 6) or
                z["enuc"].shape != (n,)):
            raise ValueError("LiH physical reference mapping mismatch")
        h1, eri, enuc = z["h1"], z["eri"], z["enuc"]
    for j, (i, R) in enumerate(zip(ref["global_rows"], ref["R_A"])):
        h = PySCFHamiltonian(compiled.sector, h1[j], eri[j], float(enuc[j]))
        ansatz = Ansatz(compiled, h, 2)
        states, values = [], []
        for theta in (ref["theta"][j], pred[j]):
            state = ansatz.state(theta)
            n2 = state @ state
            if not np.isfinite(n2) or abs(n2-1) > 1e-8:
                raise ValueError(f"Invalid LiH state norm at R={R}")
            state /= np.sqrt(n2)
            states.append(state)
            values.append(float(state @ h.apply(state)))
        ref_error = abs(values[0] - ref["energy"][j])
        if ref_error > 1e-8 or not np.isfinite(values).all():
            raise ValueError(f"LiH VQE reference replay mismatch at R={R}")
        replay_errors.append(ref_error)
        sse = float(err_theta[j] @ err_theta[j])
        old_sse = float(err_old_theta[j] @ err_old_theta[j])
        per_geometry.append(dict(
            global_row=int(i), R_angstrom=float(R),
            E_continuous_VQE_Eh=values[0], E_pred_Eh=values[1],
            energy_error_mHa=(values[1]-values[0])*1000,
            fidelity_to_continuous_VQE=float(abs(states[0] @ states[1])**2),
            theta_RMSE_continuous=float(np.sqrt(sse/60)), theta_SSE_continuous=sse,
            theta_RMSE_archived=float(np.sqrt(old_sse/60)), theta_SSE_archived=old_sse,
            focus=int(round(100*R)) in FOCUS_R))
    err_e = np.array([r["energy_error_mHa"] for r in per_geometry])
    mask = np.array([r["focus"] for r in per_geometry], dtype=bool)
    if mask.sum() != 3:
        raise ValueError("The three pre-identified LiH test geometries are missing")
    out = dict(
        n_test=len(pred), theta_RMSE_continuous=float(np.sqrt(np.mean(err_theta**2))),
        theta_RMSE_archived=float(np.sqrt(np.mean(err_old_theta**2))),
        energy_RMSE_mHa=float(np.sqrt(np.mean(err_e**2))),
        energy_MAE_mHa=float(np.mean(np.abs(err_e))),
        energy_max_abs_mHa=float(np.max(np.abs(err_e))),
        fidelity_to_VQE_min=min(r["fidelity_to_continuous_VQE"] for r in per_geometry),
        focus_energy_RMSE_mHa=float(np.sqrt(np.mean(err_e[mask]**2))),
        focus_theta_RMSE_continuous=float(np.sqrt(np.mean(err_theta[mask]**2))),
        focus_theta_SSE_continuous=float(np.sum(err_theta[mask]**2)),
        focus_theta_SSE_archived=float(np.sum(err_old_theta[mask]**2)),
        max_reference_replay_error_Eh=max(replay_errors))
    return out, per_geometry, pred
