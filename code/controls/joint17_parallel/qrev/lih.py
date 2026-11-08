"""Create one LiH-only comparison using the parent pilot's geometry split.

Preparation reads source labels to partition/write split arrays. Training and
validation do not open test.npz. This is not an externally unseen dataset.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .data import (load_raw,fit_scaler,encode_features,NUMERIC,SYSTEMS,FEATURES,PARAM)
from .common import VERSION,sha256,obj_hash,load_json,dump_json,new_dir,write_tsv

TASKS=['ucc/LiH','uccg/LiH','1kupccgsd/LiH','2kupccgsd/LiH','3kupccgsd/LiH']

def prepare_lih(data_root,parent_prepared,out):
    start=time.perf_counter();root=Path(data_root).resolve();parent=Path(parent_prepared).resolve()
    pm=load_json(parent/'manifest.json')
    for f in ('samples.tsv','schema.json','source_inventory.tsv','task_split_counts.tsv'):
        if sha256(parent/f)!=pm['files_sha256'][f]: raise ValueError('Parent metadata changed: '+f)
    if pm['split']['sample_counts']!={'train':2777,'val':605,'test':605}:
        raise ValueError('Expected the audited parent pilot split')
    source_lock=load_json(Path(__file__).resolve().parent.parent/'configs/source_lock.json')
    raw,meta,allparams,inventory,tasks,warnings=load_raw(root,source_lock)
    if obj_hash(inventory)!=pm['source_fingerprint']:
        raise ValueError('Raw tables are not the parent pilot data')
    original=pd.read_csv(parent/'samples.tsv',sep='\t',float_precision='round_trip')
    if meta.sample_id.tolist()!=original.sample_id.tolist():
        raise ValueError('Parent sample order/source identity differs')
    if not np.array_equal(meta.bondlengths.to_numpy(),original.bondlengths.to_numpy()):
        raise ValueError('Parent geometry metadata differs')
    keep=meta.task.isin(TASKS).to_numpy();parent_rows=np.flatnonzero(keep)
    subraw=raw.loc[keep].copy().reset_index(drop=True)
    submeta=meta.loc[keep].copy().reset_index(drop=True)
    submeta['parent_global_row']=parent_rows
    submeta['global_row']=np.arange(len(submeta))
    submeta['split']=original.loc[keep,'split'].to_numpy()
    if len(submeta)!=1455 or (subraw.bondlengths<=0).any():
        raise ValueError('Expected five positive-distance LiH tables, 291 rows each')
    used=[c for c in allparams if subraw[c].notna().any()]
    y=subraw[used].to_numpy(dtype=np.float64,copy=True);mask=np.isfinite(y)
    idx={s:np.flatnonzero(submeta.split.to_numpy()==s) for s in ('train','val','test')}
    if not mask[idx['train']].any(axis=0).all():raise ValueError('Unobserved train target column')
    groups={s:set(submeta.iloc[v].geometry_group) for s,v in idx.items()}
    if any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Geometry overlap between splits')
    scaler=fit_scaler(subraw[NUMERIC].to_numpy(float),idx['train'])
    schema={'version':VERSION,'numeric_features':NUMERIC,'systems':SYSTEMS,
        'input_features':FEATURES,'input_dim':14,'output_dim':len(used),
        'parameter_columns':used,'parameter_suffix_indices':[int(PARAM.fullmatch(c)[1]) for c in used],
        'scaler':scaler,'task_schemas':{k:tasks[k] for k in TASKS},
        'representative_scope':'LiH only, five ansatz tasks. All three models trained from scratch.',
        'graph_definition':{'nodes':['Li','H'],'atomic_numbers':[3,1],
            'edges':[[0,1],[1,0]],'distance':'original bondlengths; no inferred angle/units',
            'rbf_input':'shared train-only standardized bondlengths',
            'additional_information':'constant species/topology only; all models share the same 14 inputs'},
        'target_policy':'Observed non-NaN labels only; valid zeros retained; no periodic wrap; '
                        'missing full-circuit amplitudes NOT inferred.'}
    x=encode_features(subraw,schema)
    out=new_dir(out,[root,parent,root.parent/'train'])
    for s,ii in idx.items():
        np.savez_compressed(out/(s+'.npz'),X=x[ii],Y=y[ii],mask=mask[ii],global_indices=ii)
    submeta.to_csv(out/'samples.tsv',sep='\t',index=False,float_format='%.17g')
    dump_json(out/'schema.json',schema)
    inv=[r for r in inventory if r['relative_path'].endswith('_LiH_data.txt')]
    write_tsv(out/'source_inventory.tsv',inv)
    rows=[]
    for task in TASKS:
        take=submeta.task==task
        rows.append({'task':task,**{s:int((take&(submeta.split==s)).sum()) for s in idx},
            'stored_parameters':int(submeta.loc[take,'valid_label_count'].iloc[0]),
            'declared_no_parameters':int(submeta.loc[take,'no_parameters'].iloc[0])})
    write_tsv(out/'task_split_counts.tsv',rows)
    files=['train.npz','val.npz','test.npz','samples.tsv','schema.json','source_inventory.tsv','task_split_counts.tsv']
    manifest={'version':VERSION,'status':'PREPARED','data_root':str(root),'rows':len(submeta),
        'input_dim':14,'output_dim':len(used),'source_fingerprint':obj_hash(inv),'source_inventory':inv,
        'parent_prepared':str(parent),'parent_manifest_sha256':sha256(parent/'manifest.json'),
        'parent_source_fingerprint':pm['source_fingerprint'],'h6_included':False,
        'input_energy_columns_used':False,'input_task_id_used':False,
        'split':{'method':'LiH restriction of fixed parent grouped split, no resampling',
                 'sample_counts':{s:len(i) for s,i in idx.items()},
                 'geometry_group_counts':{s:len(g) for s,g in groups.items()},
                 'interpretation':'Representative within-range validation; not full 17-task comparison, not broad OOD'},
        'warnings':warnings,'scaler_train_indices_hash':scaler['fit_global_indices_sha256'],
        'files_sha256':{f:sha256(out/f) for f in files},'prepare_seconds':time.perf_counter()-start}
    dump_json(out/'manifest.json',manifest)
    return manifest
