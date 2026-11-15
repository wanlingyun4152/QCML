"""Fixed coordinate recipes for the audited six-molecule STO-3G scan."""
from pathlib import Path
from math import cos, sin, radians
import numpy as np

from .common import sha256
from .data import load_prepared

MANIFEST_SHA = "9a86fe733ab1849921bd3d51d84e0aaecbd2758c11123a567251bf816929b12d"
RULES_VERSION = "legacy_sto3g_scan_geometry_v1"
DESCRIPTOR_DIM = 14
MAX_ATOMS = 4
GRAPH_DIM = DESCRIPTOR_DIM + MAX_ATOMS + MAX_ATOMS**2
ELEMENTS = {"H2": (1, 1), "HeH+": (2, 1), "LiH": (3, 1), "H4": (1, 1, 1, 1),
            "BeH2": (4, 1, 1), "H2O": (8, 1, 1)}


def coordinates(system, r):
    """Return atomic numbers and Cartesian positions in angstrom; R is the archived scan parameter."""
    if not np.isfinite(r) or r <= 0:
        raise ValueError("Invalid scan bond length")
    z = ELEMENTS[system]
    if len(z) == 2:
        xyz = [(0, 0, 0), (0, 0, r)]
    elif system == "H4":
        xyz = [(0, 0, j*r) for j in range(4)]
    elif system == "BeH2":
        xyz = [(0, 0, 0), (0, 0, -r), (0, 0, r)]
    else:
        a = radians(104.5/2)
        x, y = r*sin(a), r*cos(a)
        xyz = [(0, 0, 0), (x, 0, y), (-x, 0, y)]
    return np.asarray(z, dtype=np.int64), np.asarray(xyz, dtype=np.float64)


def graph_features(system, r):
    atoms, xyz = coordinates(system, r)
    n = len(atoms)
    z = np.zeros(MAX_ATOMS, dtype=np.float32)
    d = np.zeros((MAX_ATOMS, MAX_ATOMS), dtype=np.float32)
    z[:n] = atoms
    dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    if np.any(dist[np.triu_indices(n, 1)] <= 0):
        raise ValueError("Coincident real atoms")
    d[:n, :n] = dist
    return np.concatenate([z, d.ravel()])


def load_joint_data(prepared, model):
    """Keep the original Y, masks, scaling and split; add geometry only to the GNN input."""
    p = Path(prepared).expanduser().resolve()
    if sha256(p/"manifest.json") != MANIFEST_SHA:
        raise ValueError("Not the exact recovered 17-task manifest; use QCML_Legacy_Exact_Recovery/prepared")
    manifest, schema, meta, data = load_prepared(p, ("train", "val"))
    if schema["input_dim"] != DESCRIPTOR_DIM or schema["output_dim"] != 303:
        raise ValueError("Unexpected original descriptor/label dimensions")
    if set(meta.system) != set(ELEMENTS) or len(set(meta.task)) != 17:
        raise ValueError("Unexpected original systems or tasks")
    if not (meta.groupby("geometry_group").split.nunique() == 1).all():
        raise ValueError("A geometry group appears in multiple splits")
    for split, n, groups, valid in (("train", 2777, 934, 172760), ("val", 605, 204, 37689)):
        a = data[split]
        row = meta.iloc[a["global_indices"]]
        if len(row) != n or row.geometry_group.nunique() != groups or int(a["mask"].sum()) != valid:
            raise ValueError("Unexpected frozen counts in " + split)
        if not np.array_equal(row.valid_label_count.to_numpy(), a["mask"].sum(axis=1)):
            raise ValueError("Target mask does not match row metadata")
        if model == "molecular_gnn":
            extras = np.stack([graph_features(r.system, r.bondlengths) for r in row.itertuples()])
            a["X"] = np.concatenate([a["X"], extras], axis=1).astype(np.float32)
    if model == "molecular_gnn":
        schema["input_dim"] = GRAPH_DIM
        schema["input_features"] = list(schema["input_features"]) + [
            "Z"+str(i) for i in range(MAX_ATOMS)
        ] + ["distance_A_"+str(i)+"_"+str(j) for i in range(MAX_ATOMS) for j in range(MAX_ATOMS)]
        schema["graph_spec"] = {"rules": RULES_VERSION, "atomic_numbers": ELEMENTS, "max_atoms": MAX_ATOMS,
                                "layout": "[14 standardized legacy descriptors, 4 Z, 4x4 distance matrix]",
                                "distance_unit": "angstrom", "padding_atomic_number": 0}
    return manifest, schema, meta, data
