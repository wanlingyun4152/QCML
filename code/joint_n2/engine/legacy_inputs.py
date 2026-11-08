"""Check data identity against the archived 17-task run, not manifest formatting."""
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
LOCK = BASE / "legacy_reference"
EXPECTED = {"train": 2777, "val": 605, "test": 605}
ARRAYS = tuple(s+".npz" for s in EXPECTED)
DATA_FILES = ("schema.json", "samples.tsv", *ARRAYS)
AUDIT_MANIFEST_SHA = "9a86fe733ab1849921bd3d51d84e0aaecbd2758c11123a567251bf816929b12d"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_lock():
    path = LOCK/"manifest.json"
    if sha(path) != AUDIT_MANIFEST_SHA:
        raise ValueError("The bundled reference lock was modified; unpack a clean copy")
    m = read_json(path)
    if sha(LOCK/"schema.json") != m["files_sha256"]["schema.json"]:
        raise ValueError("The bundled reference schema was modified")
    return m


def check_provenance(actual, ref):
    checks = ("version", "status", "rows", "input_dim", "output_dim", "h6_included",
              "source_fingerprint", "source_inventory", "scaler_train_indices_hash",
              "input_energy_columns_used", "input_task_id_used")
    problems = [k for k in checks if actual.get(k) != ref[k]]
    split = actual.get("split")
    if not isinstance(split, dict):
        problems.append("split")
    else:
        for k in ("seed", "val_fraction_of_groups", "test_fraction_of_groups", "sample_counts",
                  "geometry_group_counts", "geometry_overlap_train_val",
                  "geometry_overlap_train_test", "geometry_overlap_val_test"):
            if split.get(k) != ref["split"][k]:
                problems.append("split."+k)
    if problems:
        raise ValueError("Legacy manifest data/split differs from the original 17-task audit: "
                         + ", ".join(problems))


def check_schema(candidate, ref):
    """Require identical feature/output semantics; tolerate only scaler roundoff."""
    c, r = dict(candidate), dict(ref)
    cs, rs = dict(c.pop("scaler")), dict(r.pop("scaler"))
    if c != r:
        raise ValueError("Legacy input/parameter schema differs from the audit")
    for k in ("mean", "variance", "scale"):
        a, b = np.asarray(cs.pop(k), dtype=float), np.asarray(rs.pop(k), dtype=float)
        if a.shape != b.shape or not np.allclose(a,b,rtol=1e-12,atol=1e-13):
            raise ValueError("Rebuilt legacy scaler differs from the audit: "+k)
    if cs != rs:
        raise ValueError("Rebuilt legacy scaler fit rows or convention differ from the audit")


def verify_prepared(root, splits):
    if root is None:
        raise ValueError("--legacy-prepared must point to the original prepared directory")
    root = root.resolve()
    path = root/"manifest.json"
    current = read_json(path)
    ref = audit_lock()
    check_provenance(current, ref)
    expected = {k: ref["files_sha256"][k] for k in DATA_FILES}
    used = ("schema.json","samples.tsv",*(s+".npz" for s in splits))
    present = {k: sha(root/k) for k in used}
    bad = [k for k in used if present[k] != expected[k]]
    origin = "audited_files"
    # Only the explicitly invoked raw-table-locked rebuild emits this receipt.
    receipt_path = root/"rebuild_verified.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if (receipt.get("audited_manifest_sha256") != AUDIT_MANIFEST_SHA
                or receipt.get("source_fingerprint") != ref["source_fingerprint"]
                or receipt.get("method") != "audited_qrev_prepare_from_locked_raw_tables_v1"):
            raise ValueError("Wrong legacy rebuild receipt")
        expected = receipt.get("files_sha256", {})
        if (set(expected) != set(DATA_FILES)
                or any(current["files_sha256"].get(k) != expected[k] for k in DATA_FILES)
                or any(present[k] != expected[k] for k in used)):
            raise ValueError("Rebuilt legacy data changed after verification")
        check_schema(read_json(root/"schema.json"),read_json(LOCK/"schema.json"))
        origin = "locked_raw_rebuild"
    elif bad:
        detail = "; ".join(f"{k}: actual={present[k]}, audited={expected[k]}" for k in bad)
        raise ValueError("Legacy files differ from the archived data bytes. "
                         "Run the documented locked rebuild from the original 17 tables "
                         "or point LEGACY_PREPARED to the audited copy. "+detail)
    return dict(origin=origin, files_sha256=expected,
                source_fingerprint=ref["source_fingerprint"], manifest_sha256=sha(path))
