"""Freeze the audited 17-table dataset; explicitly exclude H6 and every unlisted table."""
import csv
import math
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .common import dump_json, load_json, sha256, obj_hash, write_tsv, new_dir, VERSION

TABLES = [('ucc','H2'),('uccg','H2'),('ucc','HeH+'),('uccg','HeH+'),
          ('ucc','LiH'),('uccg','LiH'),('uccg','H4'),('uccg','BeH2'),
          ('uccg','H2O'),('1kupccgsd','LiH'),('2kupccgsd','LiH'),
          ('3kupccgsd','LiH'),('2kupccgsd','H4'),('3kupccgsd','H4'),
          ('2kupccgsd','BeH2'),('3kupccgsd','BeH2'),('3kupccgsd','H2O')]
NUMERIC = ['bondlengths','no_qubits','no_parameters','no_gates',
           'no_Pauli_strings','no_electron','HOMO (eV)','LUMO (eV)']
SYSTEMS = sorted(set(x[1] for x in TABLES))
FEATURES = NUMERIC + ['system_' + s for s in SYSTEMS]
PARAM = re.compile(r'^VQE_param([0-9]+)$')

def load_raw(root, source_lock=None, allow_source_change=False):
    root = Path(root).expanduser().resolve()
    frames, provenance, inventory, parameters, schema, warnings = [], [], [], [], {}, []
    for ansatz, mol in TABLES:
        rel = '%s/%s_%s_data.txt' % (ansatz, ansatz, mol)
        p = root/rel
        if not p.is_file() or p.stat().st_size > 64*1024*1024:
            raise ValueError('Missing/oversized table: %s' % p)
        before = sha256(p)
        with p.open(encoding='utf8', newline='') as h:
            header = next(csv.reader(h, delimiter='\t'))
        if len(header) != len(set(header)):
            raise ValueError('Duplicate column headers: %s' % rel)
        d = pd.read_csv(p, sep='\t')
        for c in ['system']+NUMERIC:
            if c not in d.columns:
                raise ValueError('Missing column %r in %s' % (c,rel))
        if not len(d) or set(d['system'].astype(str)) != {mol}:
            raise ValueError('Unexpected system/empty table: %s' % rel)
        if not np.isfinite(d[NUMERIC].to_numpy(dtype=np.float64)).all():
            raise ValueError('Nonfinite input feature: %s' % rel)
        cols = [c for c in d.columns if PARAM.fullmatch(c)]
        if not cols or any('VQE_param' in c and c not in cols for c in d.columns):
            raise ValueError('Unexpected parameter-column convention: %s' % rel)
        y = d[cols].to_numpy(dtype=np.float64)
        if np.isinf(y).any() or (np.isfinite(y).sum(1)==0).any():
            raise ValueError('Invalid parameter labels: %s' % rel)
        if (d['no_parameters'].to_numpy(float) < np.isfinite(y).sum(1)).any():
            raise ValueError('Stored label count exceeds declared parameter count: %s' % rel)
        for c in cols:
            if c not in parameters:
                parameters.append(c)  # Preserve original first-occurrence output order.
        task = ansatz + '/' + mol
        schema[task] = {'parameter_columns_in_source_order':cols,
            'declared_parameter_counts': sorted(d['no_parameters'].unique().astype(int).tolist()),
            'stored_columns':len(cols),
            'missing_amplitudes_policy':'NOT inferred or filled into a quantum circuit'}
        if (np.isfinite(y).sum(1) != d['no_parameters'].to_numpy(float)).any():
            warnings.append('%s: no_parameters differs from stored valid label count; retained both' % task)
        frames.append(d)
        for i in range(len(d)):
            provenance.append({'sample_id':rel+':row%07d'%i, 'source_file':rel,
                               'source_row':i,'ansatz':ansatz,'task':task})
        if sha256(p) != before:
            raise ValueError('Table changed while reading: %s' % rel)
        entry = {'relative_path':rel,'sha256':before,'rows':len(d),
                 'stored_parameter_columns':len(cols),'bytes':p.stat().st_size}
        if source_lock and (rel not in source_lock or source_lock[rel]['sha256'] != before):
            if not allow_source_change:
                raise ValueError('Source differs from audited upload: %s. Review first; '
                                 '--allow-source-change is an explicit opt-out, not a repair.' % rel)
            warnings.append('Audited hash mismatch explicitly allowed: '+rel)
        inventory.append(entry)
    raw = pd.concat(frames, ignore_index=True, sort=False)
    if len(raw) != 3987 or len(parameters) != 303:
        raise ValueError('This revision entry expects 3987 rows and 303 union columns; got %d/%d. '
                         'New basis/H6 need a separate adaptation protocol.' % (len(raw), len(parameters)))
    meta = pd.DataFrame(provenance)
    for c in ['system'] + NUMERIC + [s for s in ('FCI_energy','VQE_energy') if s in raw]:
        meta[c] = raw[c].to_numpy(copy=True)
    meta.insert(0,'global_row',np.arange(len(raw),dtype=np.int64))
    # This key is specific to the current one-dimensional scans, NOT a full geometry identifier.
    meta['geometry_group'] = [str(s)+'|R='+format(float(r),'.12g')
                             for s,r in zip(meta.system,meta.bondlengths)]
    meta['valid_label_count'] = np.isfinite(raw[parameters].to_numpy(float)).sum(1)
    return raw, meta, parameters, inventory, schema, warnings

def split_groups(meta, seed=42, val_fraction=0.15, test_fraction=0.15):
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction+test_fraction >= 0.8:
        raise ValueError('Need positive val/test fractions with val+test < 0.8')
    assignment = {}
    rng = np.random.RandomState(seed)
    for system in sorted(meta.system.unique()):
        groups = np.asarray(sorted(meta.loc[meta.system==system,'geometry_group'].unique()))
        n = len(groups)
        nv, nt = int(math.ceil(n*val_fraction)), int(math.ceil(n*test_fraction))
        if n-nv-nt < 1:
            raise ValueError('Too few geometry groups for three-way split: '+system)
        order = rng.permutation(n)
        for pos,i in enumerate(order):
            assignment[str(groups[i])] = 'test' if pos < nt else ('val' if pos < nt+nv else 'train')
    labels = np.array([assignment[g] for g in meta.geometry_group])
    indices = {name:np.flatnonzero(labels==name).astype(np.int64) for name in ('train','val','test')}
    if sorted(np.concatenate(list(indices.values())).tolist()) != list(range(len(meta))):
        raise AssertionError('Split does not form a disjoint complete partition')
    for task in meta.task.unique():
        if any(not np.any((meta.task.to_numpy()==task)&(labels==s)) for s in indices):
            raise ValueError('A task is absent in a split: '+task)
    return labels, indices

def fit_scaler(numeric, train_indices):
    x = np.asarray(numeric,dtype=np.float64)[train_indices]
    if not np.isfinite(x).all() or len(x)<2:
        raise ValueError('Invalid scaler training input')
    mean = x.mean(axis=0)
    var = np.maximum(((x-mean)**2).mean(axis=0),0.0)
    scale = np.sqrt(var)
    eps = np.finfo(np.float64).eps
    scale[var <= len(x)*eps*var+(len(x)*mean*eps)**2] = 1.0
    return {'mean':mean.tolist(),'scale':scale.tolist(),'variance':var.tolist(),
            'ddof':0,'fit_split':'train','fit_count':int(len(x)),
            'fit_global_indices_sha256':obj_hash(np.asarray(train_indices).tolist())}

def encode_features(frame, schema):
    """Never mutate caller data; labels/energies/tasks are never network inputs."""
    numeric = frame[schema['numeric_features']].to_numpy(dtype=np.float64,copy=True)
    if not np.isfinite(numeric).all():
        raise ValueError('Nonfinite numeric input')
    names = schema['systems']
    unknown = set(frame.system.astype(str))-set(names)
    if unknown:
        raise ValueError('Unknown molecule categories require adaptation: %s' % sorted(unknown))
    z = (numeric-np.array(schema['scaler']['mean']))/np.array(schema['scaler']['scale'])
    onehot = (frame.system.astype(str).to_numpy()[:,None] == np.asarray(names)[None,:]).astype(np.float64)
    x = np.concatenate([z,onehot],axis=1).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Overflow while encoding features')
    return x

def prepare(data_root, out, seed=42, val_fraction=0.15, test_fraction=0.15, allow_source_change=False):
    start = time.perf_counter()
    root = Path(data_root).expanduser().resolve()
    out = new_dir(out, [root, root.parent/'train'])
    lock_path = Path(__file__).resolve().parent.parent/'configs/source_lock.json'
    lock = load_json(lock_path)
    raw, meta, params, inv, tasks, warnings = load_raw(root,lock,allow_source_change)
    labels, indices = split_groups(meta,seed,val_fraction,test_fraction)
    meta['split'] = labels
    y = raw[params].to_numpy(dtype=np.float64,copy=True)
    mask = np.isfinite(y)
    if not mask[indices['train']].any(0).all():
        raise ValueError('Output parameters with no training labels')
    scaler = fit_scaler(raw[NUMERIC].to_numpy(float),indices['train'])
    schema = {'version':VERSION,'numeric_features':NUMERIC,'systems':SYSTEMS,'input_features':FEATURES,
              'parameter_columns':params,'parameter_suffix_indices':[int(PARAM.fullmatch(c)[1]) for c in params],
              'output_dim':len(params),'input_dim':len(FEATURES),'scaler':scaler,'task_schemas':tasks,
              'target_policy':'Only non-NaN observed columns; legitimate zero labels retained; '
              'no angle wrapping, no target normalization, no missing circuit amplitude inference.'}
    x = encode_features(raw,schema)
    # Numeric arrays only. Train/val/test labels are physically separated.
    for split,idx in indices.items():
        np.savez_compressed(out/(split+'.npz'),X=x[idx],Y=y[idx],mask=mask[idx],global_indices=idx)
    meta.to_csv(out/'samples.tsv',sep='\t',index=False,float_format='%.17g')
    dump_json(out/'schema.json',schema)
    write_tsv(out/'source_inventory.tsv',inv)
    counts = []
    for task in sorted(meta.task.unique()):
        m = meta.task == task
        counts.append({'task':task,**{s:int((m&(meta.split==s)).sum()) for s in indices},
                       'stored_parameters':int(meta.loc[m,'valid_label_count'].iloc[0]),
                       'declared_no_parameters':int(meta.loc[m,'no_parameters'].iloc[0])})
    write_tsv(out/'task_split_counts.tsv',counts)
    split_groupsets = {s:set(meta.loc[labels==s,'geometry_group']) for s in indices}
    files = ['train.npz','val.npz','test.npz','samples.tsv','schema.json','source_inventory.tsv','task_split_counts.tsv']
    manifest = {'version':VERSION,'status':'PREPARED','data_root':str(root),
                'rows':len(raw),'input_dim':len(FEATURES),'output_dim':len(params),
                'source_fingerprint':obj_hash(inv),'source_inventory':inv,
                'split':{'method':'molecule + R grouped random; stratified by molecule; '
                         'same molecule/R in different ansatz tables stays in one split',
                         'seed':seed,'val_fraction_of_groups':val_fraction,'test_fraction_of_groups':test_fraction,
                         'sample_counts':{s:len(i) for s,i in indices.items()},
                         'geometry_group_counts':{s:len(g) for s,g in split_groupsets.items()},
                         'geometry_overlap_train_val':len(split_groupsets['train']&split_groupsets['val']),
                         'geometry_overlap_train_test':len(split_groupsets['train']&split_groupsets['test']),
                         'geometry_overlap_val_test':len(split_groupsets['val']&split_groupsets['test']),
                         'interpretation':'New within-range grouped random split. NOT broad OOD, '
                         'NOT an untouched external dataset: source data were historically used.'},
                'h6_included':False,'warnings':warnings,
                'input_energy_columns_used':False,'input_task_id_used':False,
                'files_sha256':{f:sha256(out/f) for f in files},
                'prepare_seconds':time.perf_counter()-start,
                'scaler_train_indices_hash':scaler['fit_global_indices_sha256']}
    dump_json(out/'manifest.json',manifest)
    print('PREPARED',str(out),manifest['split']['sample_counts'],flush=True)
    return manifest

def load_prepared(path, splits=('train','val')):
    """Training calls this with train/val only: test.npz is NOT opened or hashed here."""
    p = Path(path).resolve()
    m = load_json(p/'manifest.json')
    if m.get('status') != 'PREPARED' or m.get('version') != VERSION:
        raise ValueError('Wrong prepared-data version/status')
    for f in ['schema.json','samples.tsv']+[s+'.npz' for s in splits]:
        if sha256(p/f) != m['files_sha256'][f]:
            raise ValueError('Prepared file modified: '+f)
    schema = load_json(p/'schema.json')
    meta = pd.read_csv(p/'samples.tsv',sep='\t',float_precision='round_trip')
    arrays = {}
    for s in splits:
        if s not in ('train','val','test'):
            raise ValueError('Unknown split '+s)
        with np.load(p/(s+'.npz'),allow_pickle=False) as z:
            a = {k:z[k].copy() for k in ('X','Y','mask','global_indices')}
        if a['X'].shape != (len(a['global_indices']),schema['input_dim']) or a['Y'].shape != a['mask'].shape:
            raise ValueError('Prepared array shape mismatch')
        if a['Y'].shape[1] != schema['output_dim'] or not np.array_equal(np.isfinite(a['Y']),a['mask']):
            raise ValueError('Prepared target mask mismatch')
        if not np.isfinite(a['X']).all() or not a['mask'].any(1).all():
            raise ValueError('Nonfinite/empty prepared sample')
        if not (meta.iloc[a['global_indices']].split==s).all():
            raise ValueError('Global index/split mismatch')
        arrays[s] = a
    return m,schema,meta,arrays
