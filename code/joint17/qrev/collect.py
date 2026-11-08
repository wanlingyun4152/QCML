"""Bounded audit pack: small metrics/code only; never include dataset or weights."""
import csv
import io
import os
import tarfile
import tempfile
from pathlib import Path
from .common import dump_json,load_json,write_tsv,sha256

# Both limits are enforced, not predictions of archive size.
CONTENT_BUDGET = 8*1024*1024
ARCHIVE_HARD_LIMIT = 12*1024*1024
PER_FILE_LIMIT = 1024*1024
ALLOWED = {'manifest.json','schema.json','source_inventory.tsv','task_split_counts.tsv',
           'status.json','config.json','inference_schema.json','environment.json',
           'code_hashes.json','batch_invariance.json','validation_metrics.json',
           'best_meta.json','last_meta.json','history.tsv','metrics.json',
           'task_metrics.tsv','sample_metrics.tsv','term_metrics.tsv',
           'TEST_RESULTS.json','test_stdout.txt','suite_status.json','suite.log'}

def _bounded_history(path):
    # Always preserve header, first 30, last 200 rows; stream input to avoid huge RAM.
    from collections import deque
    first,tail = [],deque(maxlen=200)
    with path.open('rb') as f:
        header = f.readline(8192)
        n = 0
        for line in f:
            n += 1
            if n<=30: first.append(line[:4096])
            else: tail.append(line[:4096])
    if n<=230:
        return header+b''.join(first)+b''.join(tail), 'full'
    return header+b''.join(first)+b''.join(tail), 'first30_last200_rows; total_rows=%d'%n

def collect(root,out_parent=None):
    root = Path(root).expanduser().resolve()
    if not root.is_dir(): raise ValueError('Suite root not found')
    parent = Path(out_parent).expanduser().resolve() if out_parent else root.parent
    parent.mkdir(parents=True,exist_ok=True)
    dest = Path(tempfile.mkdtemp(prefix='qcml_independent_audit_',dir=str(parent)))
    used = 0
    entries,run_rows = [],[]
    candidates = []
    for dp,dirs,names in os.walk(root,followlinks=False):
        dp=Path(dp)
        dirs[:] = [d for d in sorted(dirs) if not (dp/d).is_symlink()
                   and d not in ('__pycache__','.git','predictions','checkpoints')]
        if len(dp.relative_to(root).parts)>5:
            dirs[:]=[];continue
        for name in sorted(names):
            p=dp/name;rel=p.relative_to(root)
            if p.is_symlink(): continue
            if name in ALLOWED or ('code_snapshot' in rel.parts and p.suffix in ('.py','.json','.sh','.slurm')):
                candidates.append(p)
    # Global summaries first, logs and detailed metrics last.
    candidates.sort(key=lambda p:(p.name in ('history.tsv','term_metrics.tsv','sample_metrics.tsv'),str(p)))
    for p in candidates:
        rel = p.relative_to(root)
        bsize = p.stat().st_size
        record = {'path':str(rel),'source_bytes':bsize,'source_sha256':sha256(p),'status':'pending'}
        if p.name in ('history.tsv','suite.log'):
            data,scope = _bounded_history(p)
            record['excerpt']=scope
        elif bsize>PER_FILE_LIMIT:
            record['status']='omitted_per_file_limit';entries.append(record);continue
        else:
            data=p.read_bytes()
        if used+len(data)>CONTENT_BUDGET:
            record['status']='omitted_global_budget';entries.append(record);continue
        d=dest/'payload'/rel;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(data)
        used+=len(data);record['status']='included';record['copied_bytes']=len(data);entries.append(record)
        if p.name=='status.json':
            try:
                s=load_json(p);c=load_json(p.parent/'config.json')
                row={'run':str(rel.parent),'status':s.get('status'),'model':c.get('model'),
                     'seed':c.get('seed'),'purpose':c.get('purpose'),'loss':c['training']['loss'],
                     'epochs_completed':s.get('epochs_completed'),'best_epoch':s.get('best_epoch'),
                     'best_validation_sample_mean_MSE':s.get('best_validation_sample_mean_MSE'),
                     'trainable_parameters':s.get('trainable_parameters'),
                     'invariance_passed':s.get('invariance_passed'),
                     'test_labels_opened_by_training':s.get('test_labels_opened'),
                     'prepared_manifest_sha256':s.get('prepared_manifest_sha256'),
                     'batch_order_sha256':s.get('batch_order_sha256')}
                run_rows.append(row)
            except Exception as e:
                record['summary_error']=str(e)
    if run_rows: write_tsv(dest/'run_summary.tsv',run_rows)
    pair_checks=[]
    for seed in sorted(set(r['seed'] for r in run_rows)):
        rr=[r for r in run_rows if r['seed']==seed]
        pair_checks.append({'seed':seed,'n_runs':len(rr),
             'same_prepared_manifest':len(set(r['prepared_manifest_sha256'] for r in rr))==1,
             'same_epoch_order_stream':len(set(r['batch_order_sha256'] for r in rr))==1,
             'note':'order hashes only directly comparable at equal epochs and sample counts'})
    dump_json(dest/'collection_manifest.json',{'source_root':str(root),'files':entries,
        'copied_payload_bytes':used,'content_budget_bytes':CONTENT_BUDGET,
        'archive_hard_limit_bytes':ARCHIVE_HARD_LIMIT,'pair_checks':pair_checks,
        'excluded':'All .pt/.pth, .npz, raw source datasets, full prediction exports and arbitrary logs',
        'archive_existence_is_not_training_success':True})
    archive=Path(str(dest)+'.tar.gz');tmp=Path(str(archive)+'.tmp')
    with tarfile.open(tmp,'w:gz',compresslevel=6) as t:
        t.add(dest,arcname=dest.name)
    if tmp.stat().st_size>ARCHIVE_HARD_LIMIT:
        tmp.unlink()
        raise RuntimeError('Archive hard limit exceeded; no final archive published')
    os.replace(str(tmp),str(archive))
    print('audit_tarball = '+str(archive))
    print('audit_KiB = %.3f'%(archive.stat().st_size/1024))
    return {'archive':str(archive),'bytes':archive.stat().st_size,'run_count':len(run_rows),'pair_checks':pair_checks}
