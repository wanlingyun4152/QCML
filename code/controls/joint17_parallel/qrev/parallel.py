"""Isolated directories, source locking and bounded audit for 18 parallel runs."""
import argparse,os,sys,time,csv,json,tarfile,tempfile,subprocess,traceback,hashlib
from pathlib import Path
from .common import VERSION,sha256,load_json,dump_json,outside
CONTENT_LIMIT=24*1024**2
ARCHIVE_LIMIT=12*1024**2
FILE_LIMIT=2*1024**2
MODELS=('transformer','mlp','lih_gnn')

def package():return Path(__file__).resolve().parent.parent

def fingerprint():
    root=package();rels=['parallel.py','qcml.py','collect_interfaces.py']
    for folder in ('qrev','configs','scripts','tests'):
        rels += [str(p.relative_to(root)) for p in sorted((root/folder).glob('*'))
                 if p.is_file() and p.suffix in ('.py','.json','.sh','.slurm')]
    return {r:sha256(root/r) for r in rels}

def verify(suite):
    p=load_json(suite/'plan.json')
    if p['version']!=VERSION or p['package']!=str(package()):raise ValueError('Suite package/version mismatch')
    if p['source_hashes']!=fingerprint():raise ValueError('Code/config changed since preparation')
    if sha256(suite/'prepared/manifest.json')!=p['prepared_manifest_sha256']:raise ValueError('Prepared manifest changed')
    return p

def init(data_root,parent_prepared,out_parent):
    from .lih import prepare_lih
    parent=Path(out_parent).expanduser().resolve();data=Path(data_root).resolve();old=Path(parent_prepared).resolve()
    for protected in (data,old,old.parent,package(),data.parent/'train'):outside(parent,protected)
    parent.mkdir(parents=True,exist_ok=True)
    suite=Path(tempfile.mkdtemp(prefix='qcml_lih_parallel_',dir=parent));(suite/'logs').mkdir()
    m=prepare_lih(data,old,suite/'prepared')
    jobs=[]
    for model in MODELS:
        for loss in ('weighted','unweighted'):
            for seed in (0,1,2):
                jobs.append({'id':len(jobs),'name':f'{model}_seed{seed}_{loss}',
                             'model':model,'seed':seed,'loss':loss,'epochs':3000})
    p={'version':VERSION,'package':str(package()),'suite':str(suite),
       'parent_prepared':str(old),'source_hashes':fingerprint(),
       'prepared_manifest_sha256':sha256(suite/'prepared/manifest.json'),
       'jobs':jobs,'sample_counts':m['split']['sample_counts'],'output_dim':m['output_dim'],
       'selection_metric':'sample_mean_MSE','selection_split':'val',
       'test_evaluation_planned':False,'physics_evaluation_planned':False,
       'reuse_previous_runs':False,'scope':'LiH-only 3 models x 2 losses x 3 seeds; '
       'new fixed representative comparison, not replacement numbers for all 17 tasks'}
    dump_json(suite/'plan.json',p)
    return suite

def execute(cmd,log):
    with Path(log).open('ab',buffering=0) as f:
        p=subprocess.Popen([str(c) for c in cmd],stdout=f,stderr=subprocess.STDOUT)
        rc=p.wait()
    if rc:raise RuntimeError('Command failed (%d); log=%s'%(rc,log))

def run_one(suite,row,smoke=False):
    root=suite/'smoke' if smoke else suite/'runs';root.mkdir(exist_ok=True)
    dest=root/row['name'];epochs=1 if smoke else row['epochs'];log=suite/'logs'/(('smoke_' if smoke else '')+row['name']+'.log')
    if dest.exists():raise ValueError('Run already exists; refusing overwrite/retrain: '+str(dest))
    py=sys.executable;cli=package()/'qcml.py'
    cmd=[py,cli,'train','--prepared',suite/'prepared','--config',package()/'configs'/(row['model']+'.json'),
         '--out',dest,'--seed',row['seed'],'--epochs',epochs,'--loss',row['loss'],
         '--purpose','smoke' if smoke else 'pilot','--device','cpu','--threads','1']
    execute(cmd,log)
    execute([py,cli,'evaluate','--run',dest,'--prepared',suite/'prepared','--split','val',
             '--device','cpu','--threads','1'],log)
    s=load_json(dest/'status.json');v=load_json(dest/'evaluation_val/metrics.json')
    if not(s.get('status')=='COMPLETED' and s.get('invariance_passed') is True and
           s.get('checkpoint_reload_passed') is True and s.get('training_mode_failures')==0 and
           s.get('test_labels_opened') is False and s['epochs_completed']==epochs and v['split']=='val'):
        raise RuntimeError('Run failed acceptance: '+str(dest))
    return s

def smoke(suite):
    p=verify(suite);lock=suite/'smoke.lock';lock.mkdir()
    state={'status':'STARTED','models':[],'test_evaluated':False,'stage':'unit_tests'}
    try:
        execute([sys.executable,'-m','unittest','discover','-s',str(package()/'tests'),'-v'],suite/'logs/unit_tests.log')
        for model in MODELS:
            row=next(r for r in p['jobs'] if r['model']==model and r['seed']==0 and r['loss']=='weighted')
            state['stage']=model
            s=run_one(suite,row,True)
            state['models'].append({'model':model,'status':'COMPLETED','batch_invariance':s['invariance_passed']})
        state['status']='COMPLETED';state['stage']='completed'
    except BaseException as e:
        state.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
    finally:dump_json(suite/'smoke_status.json',state)

def task(suite,index):
    p=verify(suite)
    if load_json(suite/'smoke_status.json').get('status')!='COMPLETED':raise ValueError('Shared preflight must pass first')
    if not 0<=index<len(p['jobs']):raise ValueError('Array index out of range')
    row=p['jobs'][index];locks=suite/'task_locks';locks.mkdir(exist_ok=True);(locks/row['name']).mkdir()
    states=suite/'task_status';states.mkdir(exist_ok=True)
    st={'status':'STARTED','job':row,'test_evaluated':False,'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    try:
        s=run_one(suite,row)
        st.update(status='COMPLETED',best_epoch=s['best_epoch'],batch_order_sha256=s['batch_order_sha256'])
    except BaseException as e:
        st.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
    finally:dump_json(states/(row['name']+'.json'),st)

def collect(suite):
    p=load_json(suite/'plan.json');out=Path(tempfile.mkdtemp(prefix='qcml_lih_parallel_audit_',dir=suite.parent))
    records=[];used=0
    def add(src,rel,required=False):
        nonlocal used
        src=Path(src)
        if src.is_symlink() or not src.is_file():
            records.append({'path':rel,'status':'NOT_PRESENT_OR_SYMLINK'});return
        n=src.stat().st_size
        if n>FILE_LIMIT or used+n>CONTENT_LIMIT:
            records.append({'path':rel,'status':'BUDGET_SKIP','source_bytes':n})
            if required:raise RuntimeError('Required evidence budget exceeded: '+rel)
            return
        b=src.read_bytes()
        if len(b)!=n:raise ValueError('File changed during collection')
        target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b);used+=n
        records.append({'path':rel,'status':'INCLUDED_FULL','bytes':n,'sha256':hashlib.sha256(b).hexdigest()})
    for rel in ('plan.json','smoke_status.json','submission.json'):add(suite/rel,rel,True)
    for rel in ('manifest.json','schema.json','source_inventory.tsv','task_split_counts.tsv'):
        add(suite/'prepared'/rel,'prepared/'+rel,True)
    statuses=[];orders={};accepted=True
    for row in p['jobs']:
        name=row['name'];d=suite/'runs'/name;f=suite/'task_status'/(name+'.json')
        s=load_json(f) if f.is_file() else {'status':'NOT_FINISHED_OR_NOT_SUBMITTED'}
        statuses.append({'run':name,'status':s['status']})
        add(f,'task_status/'+name+'.json',True)
        for rel in ('status.json','config.json','environment.json','code_hashes.json','best_meta.json',
                    'last_meta.json','validation_metrics.json','batch_invariance.json','history.tsv',
                    'evaluation_val/metrics.json','evaluation_val/task_metrics.tsv'):
            add(d/rel,'runs/'+name+'/'+rel,True)
        if s['status']!='COMPLETED':accepted=False
        elif row['seed'] in orders and orders[row['seed']]!=s['batch_order_sha256']:
            raise ValueError('Same-seed training row order differs')
        else:orders[row['seed']]=s.get('batch_order_sha256')
    for row in p['jobs']:
        name=row['name']
        for rel in ('evaluation_val/sample_metrics.tsv','evaluation_val/term_metrics.tsv'):
            add(suite/'runs'/name/rel,'runs/'+name+'/'+rel)
    for rel in p['source_hashes']:add(package()/rel,'source_snapshot/'+rel)
    for log in sorted((suite/'logs').glob('*.log')):add(log,'logs/'+log.name)
    summary={'status':'COMPLETED' if accepted else 'INCOMPLETE_OR_FAILED','runs':statuses,
             'test_evaluated':False,'energy_evaluated':False,
             'source_unchanged':p['source_hashes']==fingerprint()}
    dump_json(out/'collection_status.json',summary)
    dump_json(out/'collection_manifest.json',{'files':records,'copied_bytes':used,
                    'payload_limit':CONTENT_LIMIT,'archive_limit':ARCHIVE_LIMIT})
    archive=Path(str(out)+'.tar.gz');temp=Path(str(archive)+'.partial')
    with tarfile.open(temp,'w:gz',compresslevel=6) as t:t.add(out,arcname=out.name)
    if temp.stat().st_size>ARCHIVE_LIMIT:
        temp.unlink();raise ValueError('Archive over 12 MiB; final archive not published')
    temp.rename(archive)
    print('audit_tarball = '+str(archive));print('audit_KiB = %.3f'%(archive.stat().st_size/1024))
    print('collection_status = '+summary['status'])
    return archive

def main():
    a=argparse.ArgumentParser(description=__doc__);sub=a.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('init');p.add_argument('--data-root',required=True);p.add_argument('--parent-prepared',required=True);p.add_argument('--out-parent',required=True)
    for name in ('smoke','task','collect'):
        p=sub.add_parser(name);p.add_argument('--suite',required=True)
        if name=='task':p.add_argument('--index',required=True,type=int)
    x=a.parse_args()
    if x.cmd=='init':print(init(x.data_root,x.parent_prepared,x.out_parent))
    else:
        suite=Path(x.suite).resolve()
        if x.cmd=='smoke':smoke(suite)
        elif x.cmd=='task':task(suite,x.index)
        else:collect(suite)
