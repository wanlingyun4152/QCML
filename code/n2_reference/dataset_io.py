"""Read only the requested split, with data identity and shape checks."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def meta_at(root):
    meta = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if meta["schema"] != "qcml_n2_local_training_v1" or meta["n_parameters"] != 135:
        raise ValueError("Dataset schema mismatch")
    return meta


def arrays(root, split, meta):
    p = root/f"{split}.npz"
    if sha(p) != meta["sha256"][p.name]:
        raise ValueError(f"{split} checksum mismatch")
    with np.load(p, allow_pickle=False) as z:
        idx, x, y = z["index"], z["x"], z["theta"]
    if (x.shape != (len(idx), 6) or y.shape != (len(idx), 135) or
            len(idx) != meta["counts"][split] or not np.isfinite(x).all() or
            not np.isfinite(y).all() or len(np.unique(idx)) != len(idx)):
        raise ValueError(f"Bad {split} arrays")
    return idx, x, y


def scaled(x, meta):
    s = meta["input_scaling"]
    return ((x-np.asarray(s["offset"]))/np.asarray(s["scale"])).astype(np.float32)

