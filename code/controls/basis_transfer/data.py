"""Locked source/target dataset interface. The ML test arrays are never opened."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
from models import FEATURES

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):
    p=Path(path)
    if p.stat().st_size>8*1024**2:raise ValueError('Oversize JSON')
    return json.loads(p.read_text())

def check(cond,msg):
    if not cond:raise ValueError(msg)

def safe_parent(path,protected):
    p=Path(path).expanduser().resolve()
    for r in protected:
        q=Path(r).expanduser().resolve()
        check(p!=q and q not in p.parents,'Output inside protected input: '+str(q))
    p.mkdir(parents=True,exist_ok=True)
    return p

def read_index(path):
    with Path(path).open() as f:return list(csv.DictReader(f,delimiter='\t'))

def load_split(root,basis_index,split,manifest):
    check(split in ('train','val'),'Test arrays are locked in this package')
    rel='basis_%d/%s.npz'%(basis_index,split);p=Path(root)/rel
    check(not p.is_symlink() and p.is_file(),'Missing/linked data '+str(p))
    check(sha(p)==manifest['files'][rel],'Dataset byte hash mismatch '+rel)
    check(p.stat().st_size<16*1024**2,'Unexpected data size')
    with np.load(p,allow_pickle=False) as z:
        check(set(z.files)=={'X_raw','theta','sample_id','R_angstrom','feature_names'},'NPZ schema changed')
        a={k:np.array(z[k],copy=True) for k in z.files}
    n={'train':142,'val':30}[split];d=[44,434,629][basis_index]
    check(a['X_raw'].shape==(n,6) and a['theta'].shape==(n,d),'Dataset shape changed')
    check(a['feature_names'].tolist()==FEATURES,'Feature order changed')
    check(np.isfinite(a['X_raw']).all() and np.isfinite(a['theta']).all(),'Nonfinite labels/features')
    check(np.array_equal(a['X_raw'][:,0],a['R_angstrom']),'Geometry/features differ')
    check(np.all(np.diff(a['R_angstrom'])>0),'Rows must be in increasing R order')
    check(np.all(a['X_raw'][:,1]==[6,16,19][basis_index]) and
          np.all(a['X_raw'][:,2]==4) and np.all(a['X_raw'][:,3]==d),'Problem dimensions changed')
    ix=read_index(p.with_name(split+'_index.tsv'))
    check(len(ix)==n,'Index length mismatch')
    for i,row in enumerate(ix):
        check(str(a['sample_id'][i])==row['sample_id'] and float(row['R_angstrom'])==a['R_angstrom'][i],'Row identity mismatch')
    return a

def load_data(root,lock):
    root=Path(root).expanduser().resolve();mp=root/'manifest.json'
    check(sha(mp)==lock['manifest_sha256'],'This package requires the reviewed regauged manifest')
    man=read(mp);sp=read(root/'split.json')
    check(man['status']=='READY_FOR_GAUGE_REVIEW' and man['coordinate_protocol']['version']=='stage4_two_chart_regauge_v1_1_pullback','Coordinate policy mismatch')
    check(man['feature_names']==FEATURES and len(man['source_results'])==606,'Manifest schema/count mismatch')
    check(man['files']==lock['files'],'File locks changed')
    # Validate metadata/index/maps for every split, but NEVER open/hash test.npz.
    for rel,h in man['files'].items():
        if rel.endswith('.npz'):continue
        p=root/rel
        check(not p.is_symlink() and sha(p)==h,'Metadata hash mismatch '+rel)
    check(sp['counts']=={'train':142,'val':30,'test':30},'Unexpected split sizes')
    result={}
    for b in range(3):
        result[b]={s:load_split(root,b,s,man) for s in ('train','val')}
        train_r=set(map(float,result[b]['train']['R_angstrom']))
        val_r=set(map(float,result[b]['val']['R_angstrom']))
        test_r={float(k) for k,v in sp['assignment'].items() if v=='test'}
        check(not(train_r&val_r or train_r&test_r or val_r&test_r),'Split overlap')
        for s in ('train','val'):
            expected=sorted(float(k) for k,v in sp['assignment'].items() if v==s)
            check(expected==result[b][s]['R_angstrom'].tolist(),'Split does not match locked assignment')
        for n in [8,16,32,64,128,142]:
            keys=sp['nested_training_sets'][str(n)]
            check(len(keys)==n and len(set(keys))==n and set(map(float,keys))<=train_r,'Invalid nested training subset')
    for b in (1,2):
        for s in ('train','val'):check(np.array_equal(result[0][s]['R_angstrom'],result[b][s]['R_angstrom']),'Cross-basis split mismatch')
    order=list(map(float,sp['nested_training_order']))
    check(len(order)==142 and len(set(order))==142,'Bad nesting order')
    for n in [8,16,32,64,128,142]:check(set(order[:n])==set(map(float,sp['nested_training_sets'][str(n)])),'Nesting changed')
    return man,sp,result

def subset(data,keys):
    positions={float(x):i for i,x in enumerate(data['R_angstrom'])}
    idx=np.array(sorted(positions[float(k)] for k in keys),dtype=int)
    return {k:(v[idx].copy() if k!='feature_names' else v.copy()) for k,v in data.items()}
