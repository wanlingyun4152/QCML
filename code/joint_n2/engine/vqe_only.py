"""N2 physical evaluation using only matched UCCSD-VQE references."""
import csv
import json
from pathlib import Path
import sys

import numpy as np


def n2_data(root, splits):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    from dataset_io import arrays, meta_at, scaled, sha
    path = root / "dataset/uccsd_53"
    meta = meta_at(path)
    if sha(path / "manifest.json") != "5f4808c5706281571f89626f109b1205f128777d4611889c3efead9406acd883":
        raise ValueError("Unexpected N2 dataset manifest")
    result = {}
    for split in splits:
        i, x, y = arrays(path, split, meta)
        result[split] = dict(index=i, raw=x, x=scaled(x, meta), y=y, mask=np.ones(y.shape, dtype=bool))
    seen = np.concatenate([a["index"] for a in result.values()])
    if len(np.unique(seen)) != len(seen):
        raise ValueError("N2 split overlap")
    return result


def evaluate_n2(root, test, predictions):
    root = Path(root).resolve()
    sys.path.insert(0, str(root / "reference"))
    from dataset_io import meta_at, sha
    from n2_benchmark import setup_threads
    from qsector.algebra import pool
    from qsector.ansatz import Ansatz
    from qsector.sector import CompiledPool, PySCFHamiltonian, Sector
    setup_threads()
    source = root / "runs/uccsd_dense"
    data = root / "dataset/uccsd_53"
    meta = meta_at(data)
    map_path, geo_path = data / "parameter_map.json", data / "geometry_manifest.json"
    for path in (map_path, geo_path):
        if sha(path) != meta["sha256"][path.name]:
            raise ValueError(f"Hash mismatch: {path.name}")
    if sha(source / "config.json") != meta["source_config_sha256"]:
        raise ValueError("N2 reference source mismatch")
    ops, labels = pool(8, 5, "uccsd")
    op_map = [{"index": i, "label": label, "terms": [
        {"word": [list(q) for q in w], "coefficient": float(c)}
        for w, c in sorted(op.terms.items())]} for i, (op, label) in enumerate(zip(ops, labels))]
    if op_map != json.loads(map_path.read_text())["base_generators"]:
        raise ValueError("N2 operator map mismatch")
    refs = {r["index"]: r for r in json.loads(geo_path.read_text())}
    compiled = CompiledPool(Sector(8, 5, 5), ops)
    rows = []
    for j, i in enumerate(test["index"]):
        r = refs[int(i)]
        if r["split"] != "test" or r["R_angstrom"] != test["raw"][j, 0]:
            raise ValueError("N2 test split or geometry mismatch")
        path = source / "references" / f"r{i:02d}" / "reference.npz"
        if sha(path) != r["reference_sha256"]:
            raise ValueError(f"N2 reference hash mismatch at index {i}")
        with np.load(path, allow_pickle=False) as z:
            h = PySCFHamiltonian(compiled.sector, z["h1_active"], z["eri_active"], float(z["core_constant_Eh"]))
        a = Ansatz(compiled, h, 1)

        def replay(theta):
            v = a.state(theta)
            norm = np.linalg.norm(v)
            if not np.isfinite(norm) or abs(norm - 1) > 1e-8:
                raise ValueError("Invalid state norm")
            v /= norm
            return v, float(v @ h.apply(v))

        v_ref, e_ref = replay(test["y"][j])
        if abs(e_ref - r["E_VQE_Eh"]) > 1e-8:
            raise ValueError(f"VQE replay did not match at index {i}")
        for method, p in predictions.items():
            if p.shape != test["y"].shape or not np.isfinite(p).all():
                raise ValueError(f"Bad predictions: {method}")
            v, e = replay(p[j])
            rows.append(dict(method=method, index=int(i), R_angstrom=r["R_angstrom"],
                             E_VQE_Eh=e_ref, E_pred_Eh=e,
                             energy_error_mHa=1000*(e-e_ref),
                             theta_rmse=float(np.sqrt(np.mean((p[j]-test["y"][j])**2))),
                             fidelity_to_VQE=float(abs(v @ v_ref)**2)))
    summary = {}
    for method in predictions:
        m = [r for r in rows if r["method"] == method]
        err = np.array([r["energy_error_mHa"] for r in m])
        summary[method] = dict(energy_RMSE_mHa=float(np.sqrt(np.mean(err**2))),
                               energy_MAE_mHa=float(np.mean(np.abs(err))),
                               energy_max_abs_mHa=float(np.max(np.abs(err))),
                               theta_RMSE=float(np.sqrt(np.mean([r["theta_rmse"]**2 for r in m]))),
                               fidelity_to_VQE_min=min(r["fidelity_to_VQE"] for r in m))
    return summary, rows


def csv_out(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
