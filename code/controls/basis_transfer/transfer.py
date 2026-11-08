#!/usr/bin/env python3
"""QCML cross-basis pretraining/fine-tuning controlled comparison. No quantum solver."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,csv,datetime as dt,hashlib,json,os,platform,re,shutil,subprocess,tarfile,tempfile,traceback
import numpy as np
from data import read,sha,check,safe_parent,load_data,subset
from models import Model,features
from train import train_run,write

ROOT=Path(__file__).resolve().parent
SIZES=[8,16,32,64,128,142]
BASES=['sto-3g','6-31g*','cc-pvdz']
CODE_FILES=['models.py','data.py','train.py','transfer.py','dataset_lock.json','run_worker.slurm','collect.slurm','submit_transfer.sh','tests/test_transfer.py','tests/split_fixture.json']
ARCHIVE_CAP=12*1024**2;CONTENT_CAP=16*1024**2

def lock_code():return {k:sha(ROOT/k) for k in CODE_FILES}
def check_code(plan):check(lock_code()==plan['code_sha256'],'Training/submission code changed after prepare')
def bundles():return [{'index':i,'family':f,'seed':s} for i,(f,s) in enumerate((f,s) for f in ['transformer','mlp'] for s in [0,1,2])]

def env_info():
    import torch,inspect
    check('weights_only' in inspect.signature(torch.load).parameters,'Safe tensor loading unsupported')
    return {'python':platform.python_version(),'executable':sys.executable,'torch':str(torch.__version__),'numpy':np.__version__,'ready':True,'CPU_only':True,'no_sklearn_or_PySCF_required':True}

def prepare(dataset,out_parent,epochs=3000):
    ds=Path(dataset).expanduser().resolve();lock=read(ROOT/'dataset_lock.json')
    man,sp,arrays=load_data(ds,lock)
    parent=safe_parent(out_parent,[ds,ROOT]);suite=Path(tempfile.mkdtemp(prefix='qcml_basis_transfer_',dir=str(parent)))
    (suite/'runs').mkdir();(suite/'logs').mkdir();(suite/'workers').mkdir()
    plan={'version':'basis_transfer_stage5_v1','created_UTC':dt.datetime.now(dt.timezone.utc).isoformat(),
          'dataset':str(ds),'dataset_manifest_sha256':sha(ds/'manifest.json'),'dataset_lock':lock,
          'epochs':epochs,'source_basis':0,'target_bases':[1,2],'sizes':SIZES,'bundles':bundles(),
          'planned_source_runs':6,'planned_target_runs':144,'planned_total_runs':150,
          'input_rescaling':{'offset':[1.8,0,0,0,0,0],'scale':[1,20,4,640,1,1],
                            'fitted_on_any_data':False,'description':'fixed units/size scaling, no source/target label normalization'},
          'old_checkpoints_used':False,'new_STO3G_pretraining':True,
          'target_head_reset':True,'all_target_layers_trainable':True,
          'optimizer_state_transferred':False,'target_val_used_for_checkpoint_selection':True,
          'ML_final_test_evaluated':False,'energy_evaluated':False,'basis_invariance_claimed':False,
          'code_sha256':lock_code(),'environment_at_prepare':env_info(),
          'dataset_policy':man['coordinate_protocol'],'split':sp}
    write(suite/'plan.json',plan)
    src=suite/'source_snapshot';src.mkdir()
    for f in CODE_FILES:
        p=src/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f,p)
    shutil.copyfile(ds/'manifest.json',suite/'dataset_manifest.json')
    print('suite_dir = '+str(suite),flush=True);return suite

def run_bundle(suite,plan,b,data):
    index=b['index'];family=b['family'];seed=b['seed'];bundle_dir=Path(suite)/'runs'/('%s_seed%d'%(family,seed))
    bundle_dir.mkdir(exist_ok=False)
    src=train_run(bundle_dir/'source_sto3g',family,seed,data[0]['train'],data[0]['val'],plan['epochs'],
                  extra={'role':'source_pretrain','basis':'sto-3g','target_n':None})
    source=bundle_dir/'source_sto3g'/'best.pt';results=[src]
    for bi in (1,2):
        for n in SIZES:
            selected=subset(data[bi]['train'],plan['split']['nested_training_sets'][str(n)])
            pair=[]
            for mode in ('scratch','finetune'):
                run=bundle_dir/('basis%d_n%03d_%s'%(bi,n,mode))
                result=train_run(run,family,seed,selected,data[bi]['val'],plan['epochs'],
                                source=source if mode=='finetune' else None,
                                extra={'role':mode,'basis':BASES[bi],'target_n':n,'parent_source_run':str(source.parent) if mode=='finetune' else None})
                pair.append(result);results.append(result)
            check(pair[0]['initial_head_sha256']==pair[1]['initial_head_sha256'],'Paired heads differ')
            check(pair[0]['train_order_sha256']==pair[1]['train_order_sha256'],'Paired training order differs')
    return results

def worker(suite,index,count):
    import torch
    suite=Path(suite).resolve();plan=read(suite/'plan.json');check_code(plan)
    check(1<=count<=6 and 0<=index<count,'Invalid worker assignment')
    torch.set_num_threads(1);env_info()
    record=suite/'workers'/('worker_%02d.json'%index)
    check(not record.exists(),'This worker has already started; do not overwrite')
    state={'status':'RUNNING','bundles':[],'assigned_bundle_indices':list(range(index,6,count)),'ML_final_test_evaluated':False}
    write(record,state)
    try:
        _,_,data=load_data(plan['dataset'],plan['dataset_lock'])
        for b in plan['bundles'][index::count]:
            try:
                out=run_bundle(suite,plan,b,data)
                state['bundles'].append({'index':b['index'],'status':'COMPLETED','runs':len(out)})
            except BaseException as e:
                state['bundles'].append({'index':b['index'],'status':'FAILED','error':repr(e),'traceback':traceback.format_exc()})
                write(record,state)
                if isinstance(e,(KeyboardInterrupt,SystemExit)):raise
            write(record,state)
        state['status']='COMPLETED' if all(x['status']=='COMPLETED' for x in state['bundles']) else 'FINISHED_WITH_ISSUES'
        # Detect accidental writes without touching any test array.
        load_data(plan['dataset'],plan['dataset_lock'])
    except BaseException as e:
        state.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());write(record,state);raise
    write(record,state)
    if state['status']!='COMPLETED':raise RuntimeError('Some training bundles failed; inspect worker/run records')

def expected_runs(plan):
    for b in plan['bundles']:
        prefix='runs/%s_seed%d/'%(b['family'],b['seed'])
        yield prefix+'source_sto3g',b,'source_pretrain',0,142
        for bi in (1,2):
            for n in SIZES:
                for mode in ('scratch','finetune'):yield prefix+'basis%d_n%03d_%s'%(bi,n,mode),b,mode,bi,n

def collect(suite):
    import statistics
    suite=Path(suite).resolve();plan=read(suite/'plan.json')
    out=Path(tempfile.mkdtemp(prefix='qcml_basis_transfer_audit_',dir=str(suite.parent)))
    allrows=[];missing=[];candidates=[]
    for rel,b,role,bi,n in expected_runs(plan):
        p=suite/rel
        rf=p/'result.json';st=p/'status.json'
        if rf.is_file() and st.is_file() and read(st).get('status')=='COMPLETED_VALIDATION_ONLY':
            d=read(rf);cfg=read(p/'config.json')
            check(d['ML_final_test_evaluated'] is False and d['energy_evaluated'] is False,'Unexpected scope')
            check(d['invariance']['passed'] and d['checkpoint_reload_passed'],'Invalid model result')
            row={'run':rel,'family':b['family'],'seed':b['seed'],'role':role,'basis':BASES[bi],
                 'n_train':n,'best_epoch':d['best_epoch'],'MAE':d['metrics']['MAE'],'RMSE':d['metrics']['RMSE'],
                 'MSE':d['metrics']['MSE'],'training_seconds':d['training_seconds'],'best_sha256':d['best_sha256'],
                 'head_sha256':d['initial_head_sha256'],'order_sha256':d['train_order_sha256'],
                 'source_sha256':d['source_checkpoint_sha256'] or ''}
            allrows.append(row)
        else:missing.append({'run':rel,'status':read(st).get('status') if st.is_file() else 'NOT_STARTED_OR_NO_STATUS'})
        for name in ['status.json','result.json','config.json','best_meta.json','initialization.json','history_sampled.tsv','validation_samples.tsv']:
            candidates.append((p/name,rel+'/'+name))
        if role=='source_pretrain':candidates.append((p/'history.tsv',rel+'/history.tsv'))
    lookup={x['run']:x for x in allrows};paired=[]
    for x in allrows:
        if x['role']!='finetune':continue
        key=x['run'].replace('_finetune','_scratch');other=lookup.get(key)
        if other:
            check(x['head_sha256']==other['head_sha256'] and x['order_sha256']==other['order_sha256'],'Pair identity failed')
            source=lookup.get(x['run'].rsplit('/',1)[0]+'/source_sto3g')
            check(source and x['source_sha256']==source['best_sha256'],'Wrong pretraining checkpoint lineage')
            paired.append({'family':x['family'],'basis':x['basis'],'n_train':x['n_train'],'seed':x['seed'],
                'scratch_RMSE':other['RMSE'],'finetune_RMSE':x['RMSE'],
                'relative_RMSE_reduction':1-x['RMSE']/other['RMSE'] if other['RMSE']>0 else None})
    group=[]
    for fam in ['transformer','mlp']:
        for bi in (1,2):
            for n in SIZES:
                for mode in ['scratch','finetune']:
                    rs=[x for x in allrows if x['family']==fam and x['basis']==BASES[bi] and x['n_train']==n and x['role']==mode]
                    if not rs:continue
                    v={'family':fam,'basis':BASES[bi],'n_train':n,'role':mode,'seeds_received':len(rs)}
                    for k in ['MAE','RMSE','MSE','training_seconds']:
                        v[k+'_mean']=statistics.mean(x[k] for x in rs)
                        v[k+'_sample_SD']=statistics.stdev(x[k] for x in rs) if len(rs)>1 else None
                    group.append(v)
    def table(p,rows):
        if not rows:return
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
    table(out/'runs.tsv',allrows);table(out/'groups.tsv',group);table(out/'paired.tsv',paired)
    write(out/'collection_status.json',{'status':'COMPLETED_TRAINING_AND_VALIDATION' if len(allrows)==150 else 'INCOMPLETE_OR_FAILED',
          'completed_runs':len(allrows),'expected_runs':150,'incomplete':missing,'ML_final_test_evaluated':False,
          'energy_evaluated':False,'target_history_in_archive':'sampled first5,last5,every20,best; full histories on server'})
    for name in ['plan.json','dataset_manifest.json','submission.json']:candidates.append((suite/name,name))
    candidates += [(p,'workers/'+p.name) for p in sorted((suite/'workers').glob('*.json'))]
    candidates += [(suite/'source_snapshot'/f,'source_snapshot/'+f) for f in CODE_FILES]
    size=sum(p.stat().st_size for p in out.iterdir() if p.is_file());records=[]
    for p,rel in candidates:
        item={'file':rel,'status':'MISSING'}
        if p.is_symlink():item['status']='SKIPPED_SYMLINK'
        elif p.is_file():
            n=p.stat().st_size;item.update(bytes=n,sha256=sha(p))
            if n>2*1024**2:item['status']='SKIPPED_FILE_CAP'
            elif size+n>CONTENT_CAP:item['status']='SKIPPED_BUDGET'
            else:
                q=out/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);size+=n;item['status']='INCLUDED'
        records.append(item)
    write(out/'collection_manifest.json',records)
    dest=Path(str(out)+'.tar.gz');partial=Path(str(dest)+'.partial')
    with tarfile.open(partial,'w:gz') as t:
        for p in sorted(out.rglob('*')):
            if p.is_file() and not p.is_symlink():t.add(p,arcname=out.name+'/'+str(p.relative_to(out)),recursive=False)
    if partial.stat().st_size>ARCHIVE_CAP:
        partial.unlink();raise RuntimeError('12 MiB hard cap; no archive published')
    partial.rename(dest)
    print('audit_tarball = '+str(dest));print('completed_runs = %d/150'%len(allrows));print('final_test_evaluated = false')
    return dest

def queue_count():
    p=subprocess.run(['squeue','--array','--noheader','--user',str(os.getuid()),'--format=%i'],capture_output=True,text=True,timeout=30)
    check(p.returncode==0,'squeue failed: '+p.stderr)
    lines={s.strip() for s in p.stdout.splitlines() if s.strip()}
    check(all('[' not in s and re.fullmatch(r'[0-9]+(?:_[0-9]+)?(?:\+[0-9]+)?',s) for s in lines),'Could not count expanded queue safely')
    return len(lines)

def submit(dataset,out_parent,partition='intel',limit=40,reserve=2,workers=6):
    count=queue_count();available=limit-count-reserve-1
    n=min(6,workers,available);check(n>=1,'No queue space for at least one worker plus collect and reserve')
    suite=prepare(dataset,out_parent)
    env=os.environ.copy();env.update(QCML_TRANSFER_PACKAGE=str(ROOT),QCML_TRANSFER_SUITE=str(suite),QCML_TRANSFER_WORKERS=str(n),QCML_PYTHON=sys.executable)
    base=['sbatch','--parsable','--export=ALL','--partition='+partition]
    if os.environ.get('QCML_ACCOUNT'):base+=['--account='+os.environ['QCML_ACCOUNT']]
    rec={'queue_snapshot':count,'assumed_limit':limit,'reserve':reserve,'workers':n,'submissions':[]}
    write(suite/'submission.json',rec)
    def sb(args):
        entry={'args':args,'status':'SUBMITTING'};rec['submissions'].append(entry);write(suite/'submission.json',rec)
        try:p=subprocess.run(base+args,env=env,capture_output=True,text=True,timeout=60)
        except (subprocess.TimeoutExpired,OSError) as e:
            entry.update(status='ACCEPTANCE_UNKNOWN',error=repr(e));write(suite/'submission.json',rec);raise
        if p.returncode:
            entry.update(status='REJECTED',stderr=p.stderr);write(suite/'submission.json',rec);raise RuntimeError(p.stderr)
        job=p.stdout.strip().split(';')[0]
        if not re.fullmatch('[0-9]+',job):
            entry.update(status='ACCEPTANCE_UNKNOWN',stdout=p.stdout);write(suite/'submission.json',rec);raise RuntimeError('Unknown sbatch response; inspect before resubmitting')
        entry.update(status='ACCEPTED',job=job);write(suite/'submission.json',rec);return job
    job=sb(['--array=0-%d'%(n-1),'--output='+str(suite/'logs/worker-%A_%a.log'),'--error='+str(suite/'logs/worker-%A_%a.err'),str(ROOT/'run_worker.slurm')])
    print('worker_array_job = '+job,flush=True)
    col=sb(['--dependency=afterany:'+job,'--output='+str(suite/'logs/collect-%j.log'),'--error='+str(suite/'logs/collect-%j.err'),str(ROOT/'collect.slurm')])
    print('collect_job = '+col);print('workers = %d; planned model fits = 150; no final-test/physics evaluation'%n)
    return suite

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('check-env')
    for cmd in ['prepare','submit']:
        s=sub.add_parser(cmd);s.add_argument('--dataset',required=True);s.add_argument('--out-parent',required=True)
        if cmd=='submit':
            s.add_argument('--partition',default='intel');s.add_argument('--limit',type=int,default=40);s.add_argument('--reserve',type=int,default=2);s.add_argument('--workers',type=int,default=6)
    s=sub.add_parser('worker');s.add_argument('--suite',required=True);s.add_argument('--index',type=int,required=True);s.add_argument('--count',type=int,required=True)
    s=sub.add_parser('collect');s.add_argument('--suite',required=True)
    x=p.parse_args()
    if x.cmd=='check-env':print(json.dumps(env_info(),indent=2))
    elif x.cmd=='prepare':prepare(x.dataset,x.out_parent)
    elif x.cmd=='submit':
        check(1<=x.workers<=6 and x.limit>=1 and x.reserve>=0,'Invalid queue budget');submit(x.dataset,x.out_parent,x.partition,x.limit,x.reserve,x.workers)
    elif x.cmd=='worker':worker(x.suite,x.index,x.count)
    elif x.cmd=='collect':collect(x.suite)

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();sys.exit(1)
