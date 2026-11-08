"""Small, auditable I/O utilities. Source directories are never modified."""
import csv
import hashlib
import inspect
import json
import os
import platform
import sys
from pathlib import Path

VERSION = '1.0.0'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def obj_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()

def dump_json(path, obj):
    path = Path(path)
    text = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text + '\n', encoding='utf8')
    os.replace(str(tmp), str(path))

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def write_tsv(path, rows, columns=None):
    rows = list(rows)
    if not rows and columns is None:
        raise ValueError('An empty table needs explicit columns')
    with open(path, 'w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns or list(rows[0]), delimiter='\t')
        w.writeheader()
        w.writerows(rows)

def outside(path, protected):
    p, r = Path(path).expanduser().resolve(), Path(protected).expanduser().resolve()
    if p == r or r in p.parents:
        raise ValueError('Output %s must be outside protected source %s' % (p, r))
    return p

def new_dir(path, protected=()):
    p = Path(path).expanduser().resolve()
    for r in protected:
        outside(p, r)
    p.mkdir(parents=True, exist_ok=False)
    return p

def environment():
    import numpy as np
    import pandas as pd
    import torch
    return {'python': platform.python_version(), 'python_executable': sys.executable,
            'numpy': str(np.__version__), 'pandas': str(pd.__version__),
            'torch': str(torch.__version__), 'cuda_available': bool(torch.cuda.is_available()),
            'conda_env': os.environ.get('CONDA_DEFAULT_ENV'),
            'hostname': platform.node(),
            'slurm': {k: os.environ[k] for k in ('SLURM_JOB_ID','SLURM_JOB_PARTITION',
                'SLURM_CPUS_PER_TASK','SLURM_JOB_NODELIST') if k in os.environ}}

def setup(seed, device, threads):
    import random
    import numpy as np
    import torch
    if not 1 <= threads <= 64:
        raise ValueError('threads must be 1..64')
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; refusing silent CPU fallback')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return torch.device(device)

def sync(device):
    import torch
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)

def save_weights(path, model):
    import torch
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(state, str(tmp))
    os.replace(str(tmp), str(path))
    return sha256(path)

def read_weights(path):
    import torch
    if 'weights_only' not in inspect.signature(torch.load).parameters:
        raise RuntimeError('torch.load(weights_only=True) required; no unsafe fallback')
    state = torch.load(str(path), map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected a tensor-only state_dict, not a legacy full-object checkpoint')
    if not all(torch.isfinite(v).all().item() for v in state.values() if v.is_floating_point()):
        raise ValueError('Checkpoint contains nonfinite tensors')
    return state

def source_snapshot(out):
    """Only copy the small code/config distribution, never source data or weights."""
    import shutil
    root = Path(__file__).resolve().parent.parent
    dest = Path(out) / 'code_snapshot'
    dest.mkdir()
    rows = []
    for rel in ['qcml.py'] + [str(p.relative_to(root)) for folder in ('qrev','configs','scripts')
                             for p in sorted((root/folder).glob('*')) if p.is_file()
                             and p.suffix in ('.py','.json','.sh','.slurm')]:
        src = root / rel
        d = dest / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(src), str(d))
        rows.append({'path': rel, 'sha256': sha256(d)})
    dump_json(Path(out)/'code_hashes.json', rows)
    return obj_hash(rows)
