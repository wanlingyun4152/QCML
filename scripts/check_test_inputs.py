#!/usr/bin/env python3
"""Check the delivered input-only test data; no model or reference labels needed."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT/'tests/inputs'


def load_inputs(path):
    allowed = {'features','raw_features','domain','global_indices','index','task','system','R_angstrom'}
    with np.load(path,allow_pickle=False) as archive:
        if not set(archive.files).issubset(allowed):
            raise ValueError('Unexpected field in input-only archive: '+str(path))
        data = {key:np.array(archive[key],copy=True) for key in archive.files}
    domain = str(data['domain'].item())
    if domain not in {'old','n2'}:
        raise ValueError('Unknown model domain')
    features = data['features']
    dim = {'old':14,'n2':6}[domain]
    if features.ndim!=2 or features.shape[1]!=dim or not np.isfinite(features).all():
        raise ValueError('Invalid input features')
    if data['raw_features'].shape!=features.shape or not np.isfinite(data['raw_features']).all():
        raise ValueError('Invalid physical input features')
    if data['R_angstrom'].shape!=(len(features),) or not (data['R_angstrom']>0).all():
        raise ValueError('Invalid bond coordinates')
    if domain=='old':
        if len(np.unique(data['global_indices']))!=len(features):
            raise ValueError('Repeated source row')
    elif len(np.unique(data['index']))!=len(features):
        raise ValueError('Repeated N2 index')
    return data


def main():
    manifest = json.loads((INPUTS/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files_sha256'].items():
        path = INPUTS/name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Input checksum failed: '+name)
    old = load_inputs(INPUTS/'legacy17/model_inputs.npz')
    n2 = load_inputs(INPUTS/'n2/model_inputs.npz')
    if old['features'].shape!=(605,14) or n2['features'].shape!=(9,6):
        raise ValueError('Test sample counts differ from the recorded splits')
    if len(set(old['task']))!=17:
        raise ValueError('Legacy task count changed')
    with (INPUTS/'lih/test_geometries.csv').open(newline='',encoding='utf-8') as stream:
        lih = list(csv.DictReader(stream))
    if len(lih)!=44:
        raise ValueError('LiH held-out input count changed')
    basis = json.loads((INPUTS/'basis_cases/lih_cases.json').read_text())
    if len(basis['cases'])!=3 or any(len(c['bond_lengths_angstrom'])!=30 for c in basis['cases']):
        raise ValueError('Basis input geometry counts changed')
    dynamics = json.loads((INPUTS/'dynamics/lih_initial_conditions.json').read_text())
    if np.asarray(dynamics['coordinates_bohr']).shape!=(2,3) or np.asarray(dynamics['velocities_au']).shape!=(2,3):
        raise ValueError('Invalid initial condition shape')
    print('TEST_INPUTS_OK legacy17=605 n2=9 lih=44 basis_cases=3x30 dynamics=1')


if __name__=='__main__':
    main()
