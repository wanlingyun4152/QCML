#!/usr/bin/env python3
"""One isolated legacy-AIMD timestep comparison. No fitting or force revalidation.
Python >=3.9. This controller uses stdlib; the copied original MD needs its OLD environment.
"""
import argparse
import ast
import csv
import datetime as dt
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import traceback

PACKAGE = Path(__file__).resolve().parent
MIB = 1024 * 1024
_child = None


def check(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(MIB), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_parent(parent, protected):
    p = Path(parent).expanduser().resolve()
    for q in protected:
        q = Path(q).expanduser().resolve()
        check(p != q and q not in p.parents, 'Output lies inside protected input: ' + str(q))
    return p


def copy_checked(source, target, expected=None, max_bytes=64*MIB):
    source, target = Path(source), Path(target)
    check(not source.is_symlink() and source.is_file(), 'Missing file or direct symlink: ' + str(source))
    check(source.stat().st_size <= max_bytes, 'Input exceeds copy budget: ' + str(source))
    before = source.stat()
    digest = sha(source)
    check(expected is None or digest == expected, 'SHA256 mismatch; do not bypass: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as a, target.open('xb') as b:
        shutil.copyfileobj(a, b, MIB)
    after = source.stat()
    check((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
          and sha(target) == digest, 'Source changed during copying: ' + str(source))
    return {'source': str(source.resolve()), 'copy': str(target.resolve()),
            'sha256': digest, 'bytes': after.st_size}


def resolve_sources(base, explicit=None):
    """Only the two concrete locations already evidenced in this conversation."""
    base = Path(base).expanduser().resolve()
    lock = read_json(PACKAGE/'source_lock.json')
    paths = ([Path(explicit).expanduser().resolve()] if explicit else [
        base/'md_new/lih_uccg/aimd_dipole', base/'md_new/aimd/lih_uccg_300k'])
    reports = []
    for run in paths:
        errors = []
        for item in lock['scientific_sources']:
            p = (run if item['role'] == 'run' else base/'md_new/src_force_dipole')/item['name']
            if not p.is_file() or p.is_symlink() or sha(p) != item['sha256']:
                errors.append(str(p))
        p = run/'transformer_model_params.pth'
        if not p.is_file() or p.is_symlink() or sha(p) != lock['checkpoint_sha256']:
            errors.append(str(p))
        reports.append({'candidate': str(run), 'mismatches_or_missing': errors})
        if not errors:
            return run, reports
    raise ValueError('No hash-matching runtime. Nothing will be executed.\n' + json.dumps(reports, indent=2))


def snapshot(base, run, suite):
    """Copy immutable inputs, never use hardlinks for mutable legacy work files."""
    lock = read_json(PACKAGE/'source_lock.json')
    manifest = []
    for item in lock['scientific_sources']:
        src = (run if item['role']=='run' else base/'md_new/src_force_dipole')/item['name']
        manifest.append(copy_checked(src, suite/'original_code'/item['name'], item['sha256'], MIB))
    manifest.append(copy_checked(run/'transformer_model_params.pth', suite/'inputs/transformer_model_params.pth', lock['checkpoint_sha256']))
    manifest.append(copy_checked(run/'zeroindex_LiH.txt', suite/'inputs/zeroindex_LiH.txt', max_bytes=MIB))
    total = 0
    for rel in lock['table_paths']:
        row = copy_checked(base/'data'/rel, suite/'inputs/data'/rel, max_bytes=16*MIB)
        total += row['bytes']; check(total <= 64*MIB, '17-table content exceeds 64 MiB')
        manifest.append(row)
    mo_root = base/'vqechem/lih/mo'
    check(mo_root.is_dir(), 'Missing original MO directory: '+str(mo_root))
    mo_files = sorted(mo_root.glob('mo_*.txt'))
    check(1 <= len(mo_files) <= 1000, 'Unexpected number of original MO files')
    total = 0
    for p in mo_files:
        row = copy_checked(p, suite/'inputs/mo'/p.name, max_bytes=MIB)
        total += row['bytes']; check(total <= 16*MIB, 'MO text files exceed 16 MiB')
        manifest.append(row)
    check((suite/'inputs/mo/mo_1.55.txt').is_file(), 'Missing MO for original starting geometry')
    log = base/'md_new/lih_uccg/ir/job-35435.log'
    check(log.is_file() and sha(log)==lock['baseline_log_sha256'], 'Original dt=5 log differs or is missing')
    manifest.append({'source':str(log.resolve()),'copy':None,'sha256':lock['baseline_log_sha256'],'bytes':log.stat().st_size})
    write_json(suite/'input_manifest.json', manifest)
    # All immutable copied input files become read-only; runtime scratch remains writable.
    for row in manifest:
        if row.get('copy'):
            Path(row['copy']).chmod(0o444)
    return log


def change_md(text, step, frames):
    tree = ast.parse(text)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'VelocityVerlet']
    check(len(calls)==1, 'Expected exactly one VelocityVerlet construction')
    call = calls[0]; kws={k.arg:k for k in call.keywords}
    check(set(('dt','steps','veloc','data_output','trajectory_output')) <= set(kws), 'Unexpected MD keywords')
    check(isinstance(kws['dt'].value,ast.Constant) and kws['dt'].value.value==5, 'Original dt must be 5')
    check(isinstance(kws['steps'].value,ast.Constant) and kws['steps'].value.value==2, 'Expected collected two-record driver')
    new=text
    for key,old,value in [('dt',5,step),('steps',2,frames)]:
        pattern=r'\b'+key+r'\s*=\s*'+str(old)+r'(?=\s*,)'
        new,n=re.subn(pattern,key+'='+str(value),new)
        check(n==1,'Ambiguous MD literal: '+key)
    updated=ast.parse(new)
    for t in [tree,updated]:
        c=next(n for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='VelocityVerlet')
        for k in c.keywords:
            if k.arg in ('dt','steps'): k.value=ast.Constant(value=0)
    check(ast.dump(tree)==ast.dump(updated),'Unexpected scientific AST change')
    return new


def make_case(suite, name, step, frames):
    case = suite/name
    case.mkdir(exist_ok=False)
    old = read_json(PACKAGE/'source_lock.json')['source_prefix']
    substitutions = {'nn_output.py': {old+'/data':str(suite/'inputs/data')},
                     'atom_force.py': {old+'/md_new/src_force_dipole':str(case),
                                       old+'/vqechem/lih/mo':str(suite/'inputs/mo')}}
    for value in [str(case),str(suite)]:
        check(not any(c in value for c in "'\"\n\r\\"), 'Unsafe characters in path')
    diffs=[]
    for p in sorted((suite/'original_code').glob('*.py')):
        text=p.read_bytes().decode('utf-8')
        new=text
        for a,b in substitutions.get(p.name,{}).items():
            check(new.count(a)==1,'Unexpected path occurrence: '+a)
            new=new.replace(a,b)
        if p.name=='md.py': new=change_md(new,step,frames)
        dest=case/p.name;dest.write_bytes(new.encode('utf-8'))
        diffs.append({'name':p.name,'original_sha256':sha(p),'runtime_sha256':sha(dest),
                      'changes':('only_dt_and_steps' if p.name=='md.py' else
                                 'only_explicit_path_relocation' if p.name in substitutions else 'unchanged')})
    copy_checked(suite/'inputs/transformer_model_params.pth',case/'transformer_model_params.pth')
    copy_checked(suite/'inputs/zeroindex_LiH.txt',case/'zeroindex_LiH.txt')
    write_json(case/'runtime_changes.json',diffs)
    return case


NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?'
FIELD_PATTERNS = {
    'R_angstrom':r'^\s*bondlength\s*=\s*('+NUMBER+r')\s*$',
    'E_pred_Eh':r'^\s*Predict energy\s*:\s*('+NUMBER+r')\s*$',
    'Ekin_Eh':r'^\s*Ekin\s*=\s*('+NUMBER+r')\s*$',
    'Epot_logged_Eh':r'^\s*Epot\s*=\s*('+NUMBER+r')\s*$',
    'Etot_logged_Eh':r'^\s*Etot\s*=\s*('+NUMBER+r')\s*$',
    'dipole_au':r'^\s*dipole au\s*:\s*('+NUMBER+r')\s*$',
    'dipole_Debye':r'^\s*dipole Debye\s*:\s*('+NUMBER+r')\s*$'}


def scalar(s):
    v=float(s.replace('D','E').replace('d','e'))
    check(math.isfinite(v),'Nonfinite value in MD log')
    return v


def numeric_literal(text):
    """Safely accept numeric NumPy scalar reprs, never execute a call from a log."""
    tree=ast.parse(text,mode='eval').body
    def decode(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float,complex):return n.value
        if isinstance(n,(ast.List,ast.Tuple)):return [decode(x) for x in n.elts]
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            v=decode(n.operand);return v if isinstance(n.op,ast.UAdd) else -v
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub)):
            a,b=decode(n.left),decode(n.right);return a+b if isinstance(n.op,ast.Add) else a-b
        if isinstance(n,ast.Call) and len(n.args)==1 and not n.keywords:
            f=n.func
            if isinstance(f,ast.Attribute) and isinstance(f.value,ast.Name) and f.value.id in ('np','numpy') and f.attr in ('complex128','complex64','float64','float32'):
                return decode(n.args[0])
        raise ValueError('Unsupported numeric representation in force output')
    return decode(tree)


def parse_log(path):
    """Parse each BOMD block without zipping unrelated regexp lists."""
    rows=[]; cur=None; vectors=None
    patterns={k:re.compile(p) for k,p in FIELD_PATTERNS.items()}
    def finish(block):
        missing=[k for k in FIELD_PATTERNS if k not in block]
        missing += [k for k in ('force_scalar','coord','velocity') if k not in block]
        check(not missing,'Incomplete time block '+str(block.get('time_au'))+': '+str(missing))
        check(len(block['coord'])==6 and len(block['velocity'])==6,'Incomplete geometry/velocity')
        check(not rows or block['time_au']>rows[-1]['time_au'],'Repeated/reversed time; never splice runs')
        block['Etot_pred_Eh']=block['E_pred_Eh']+block['Ekin_Eh']
        rows.append(block)
    with Path(path).open(encoding='utf-8',errors='replace') as f:
        for num,line in enumerate(f,1):
            m=re.match(r'^\s*BOMD Time\s+('+NUMBER+r')\s*$',line)
            if m:
                if cur is not None: finish(cur)
                cur={'time_au':scalar(m.group(1)),'source_line':num};vectors=None;continue
            if cur is None: continue
            for key,p in patterns.items():
                m=p.match(line)
                if m:
                    check(key not in cur,'Repeated '+key+' at time '+str(cur['time_au']))
                    cur[key]=scalar(m.group(1))
            m=re.match(r'^\s*force:\s*(.+?)\s*$',line)
            if m:
                check('force_scalar' not in cur,'Repeated force in block')
                raw=m.group(1).strip()
                if raw.startswith('['):
                    values=numeric_literal(raw)
                    check(isinstance(values,list) and len(values)==6,'Expected six gradient operator outputs')
                    z=complex(values[2])
                    cur['force_print_format']='six_component_list_Li_z_index2'
                else:
                    z=complex(raw.strip('()').replace(' ',''))
                    cur['force_print_format']='single_Li_z_component'
                check(abs(z.imag)<1e-10 and math.isfinite(z.real),'Nonreal/nonfinite force output')
                cur['force_scalar']=z.real
            if 'VelocityVerlet final geometry' in line:
                check('coord' not in cur,'Repeated final geometry');vectors='coord';cur[vectors]=[]
            elif 'VelocityVerlet final velocity' in line:
                check('velocity' not in cur,'Repeated final velocity');vectors='velocity';cur[vectors]=[]
            elif vectors:
                m=re.match(r'^\s*([01])\s+(Li|H)\s+('+NUMBER+r')\s+('+NUMBER+r')\s+('+NUMBER+r')\s*$',line)
                if m:
                    expected=len(cur[vectors])//3
                    check(int(m.group(1))==expected and m.group(2)==('Li','H')[expected],'Wrong atom row order')
                    cur[vectors].extend(scalar(m.group(i)) for i in (3,4,5))
                    if len(cur[vectors])==6:vectors=None
        if cur is not None:finish(cur)
    return rows


def prefix_comparison(rows, baseline, count=3):
    check(len(rows)>=count and len(baseline)>=count,'Insufficient prefix records')
    tolerances={'time_au':1e-12,'R_angstrom':1e-9,'E_pred_Eh':1e-9,
                'Ekin_Eh':1e-10,'Epot_logged_Eh':1e-9,'Etot_logged_Eh':1e-9,
                'force_scalar':1e-8,'dipole_au':1e-7,'dipole_Debye':3e-7,
                'coord':2e-8,'velocity':2e-10}
    tests=[]
    for i in range(count):
        for k,tol in tolerances.items():
            a,b=rows[i][k],baseline[i][k]
            d=max(abs(x-y) for x,y in zip(a,b)) if isinstance(a,list) else abs(a-b)
            tests.append({'record':i,'field':k,'max_abs':d,'atol':tol,'passed':d<=tol})
    return {'passed':all(t['passed'] for t in tests),'records':count,'checks':tests,
            'meaning':'Runtime identity check only; not force finite-difference revalidation or full-trajectory identity proof'}


def check_grid(rows, step, count):
    check(len(rows)==count,'Wrong frame count: %d vs %d'%(len(rows),count))
    for i,row in enumerate(rows):check(abs(row['time_au']-i*step)<=1e-8,'Unexpected time at row '+str(i))


def write_table(path, rows):
    columns=['time_au','R_angstrom','E_pred_Eh','Ekin_Eh','Etot_pred_Eh','Epot_logged_Eh',
             'Etot_logged_Eh','force_scalar','dipole_au','dipole_Debye','source_line']
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,delimiter='\t',extrasaction='ignore');w.writeheader();w.writerows(rows)


def energy_statistics(rows):
    check(len(rows)>1,'At least two frames required')
    times=[r['time_au'] for r in rows]; energy=[r['Etot_pred_Eh'] for r in rows]
    d=[e-energy[0] for e in energy];tmean=sum(times)/len(times);emean=sum(d)/len(d)
    slope=sum((t-tmean)*(e-emean) for t,e in zip(times,d))/sum((t-tmean)**2 for t in times)
    return {'records':len(rows),'t_start_au':times[0],'t_end_au':times[-1],
            'max_abs_change_Eh':max(map(abs,d)), 'rms_from_initial_Eh':math.sqrt(sum(x*x for x in d)/len(d)),
            'end_minus_start_Eh':d[-1],'peak_to_peak_Eh':max(d)-min(d),
            'linear_slope_Eh_per_au':slope,
            'definition':'Predict energy + same-time kinetic energy; no smoothing or detrending',
            'units':'as in unchanged historical code; time au, energy Hartree',
            'convergence_claim':False}


def execute(cmd, cwd, stdout, stderr):
    global _child
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    # No inherited PYTHONPATH may redirect local scientific imports elsewhere.
    env.pop('PYTHONPATH',None)
    with Path(stdout).open('w') as a,Path(stderr).open('w') as b:
        _child=subprocess.Popen(cmd,cwd=str(cwd),stdout=a,stderr=b,env=env)
        try:rc=_child.wait()
        finally:_child=None
    check(rc==0,'Subprocess failed (%s); see %s and %s'%(rc,stdout,stderr))


def interrupt(signum, frame):
    if _child is not None:
        _child.terminate()
        try:_child.wait(timeout=20)
        except subprocess.TimeoutExpired:_child.kill()
    raise RuntimeError('Interrupted by signal '+str(signum)+'; partial data retained, no automatic restart')


def check_environment(case, python, suite):
    # Import the actual locked runtime graph, not a guessed short list of packages.
    code = '''import sys, json, importlib, traceback
mods={}; errors=[]
for name in ['numpy','scipy','pandas','sklearn','torch','pyscf','openfermion','quspin','symmer','qiskit','projectq','pyscf.dmrgscf','integrators','nn_input','nn_output','atom_force']:
 try:
  m=importlib.import_module(name);mods[name]={'version':str(getattr(m,'__version__','unreported')),'file':getattr(m,'__file__',None)}
 except Exception as e:
  errors.append({'module':name,'error':repr(e),'traceback':traceback.format_exc()})
report={'python':sys.version,'executable':sys.executable,'modules':mods,'errors':errors,'ready':not errors}
with open(sys.argv[1],'w') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
sys.exit(0 if not errors else 2)
'''
    execute([python,'-B','-c',code,str(suite/'environment.json')],case,suite/'environment_imports.log',suite/'environment.err')
    r=read_json(suite/'environment.json')
    for name in ['integrators','nn_input','nn_output','atom_force']:
        check(Path(r['modules'][name]['file']).resolve()==(case/(name+'.py')).resolve(),'Unexpected import origin: '+name)


def verify_inputs(suite):
    checks=[]
    for row in read_json(suite/'input_manifest.json'):
        source=Path(row['source'])
        ok=source.is_file() and sha(source)==row['sha256']
        copied=True if not row.get('copy') else sha(row['copy'])==row['sha256']
        checks.append({'path':str(source),'original_unchanged':ok,'snapshot_unchanged':copied})
    write_json(suite/'input_integrity_after.json',checks)
    check(all(x['original_unchanged'] and x['snapshot_unchanged'] for x in checks),'An original or immutable input changed')


def collect(suite):
    """Bounded text audit, no weights, MO files, tables, or whole MD logs."""
    suite=Path(suite).expanduser().resolve()
    check((suite/'plan.json').is_file(),'Not a comparison suite')
    out=Path(tempfile.mkdtemp(prefix='qcml_aimd_dt25_audit_',dir=str(suite.parent)))
    files=[];total=0
    selected=list(suite.glob('*.json'))+list(suite.glob('*.tsv'))
    selected += list(suite.glob('*/runtime_changes.json'))
    selected += list(suite.glob('*/observation_provenance.json'))
    for p in sorted(set(selected)):
        size=p.stat().st_size
        check(size<=6*MIB and total+size<=16*MIB,'Audit content budget exceeded; original files retained')
        q=out/p.relative_to(suite);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
        files.append({'path':str(q.relative_to(out)),'bytes':size,'sha256':sha(q)});total+=size
    for p in sorted(list(suite.glob('**/*.err'))+list(suite.glob('**/*.log'))):
        if p.is_symlink():continue
        with p.open('rb') as f:
            head=f.read(16000);f.seek(0,2);end=f.tell();f.seek(max(0,end-32000));tail=f.read()
        q=out/'log_excerpts'/str(p.relative_to(suite)).replace('/','__')
        q.parent.mkdir(parents=True,exist_ok=True)
        q.write_bytes(head+b'\n[EXPLICIT HEAD/TAIL EXCERPT, MAY OVERLAP]\n'+tail)
        total+=q.stat().st_size;check(total<=16*MIB,'Audit content budget exceeded')
        files.append({'path':str(q.relative_to(out)),'bytes':q.stat().st_size,'sha256':sha(q),'excerpt':True})
    write_json(out/'collection_manifest.json',files)
    archive=Path(str(out)+'.tar.gz');partial=Path(str(archive)+'.partial')
    with tarfile.open(partial,'w:gz') as tf:tf.add(out,arcname=out.name)
    check(partial.stat().st_size<=8*MIB,'Archive exceeds 8 MiB; not published')
    partial.rename(archive)
    print('audit_tarball =',archive,flush=True)
    print('audit_KiB =',round(archive.stat().st_size/1024,3),flush=True)
    return archive


def run(base, source, parent, python):
    base=Path(base).expanduser().resolve()
    runtime,candidates=resolve_sources(base,source)
    protected=[runtime,base/'md_new/src_force_dipole',base/'data',base/'vqechem/lih/mo',PACKAGE]
    parent=safe_parent(parent,protected);parent.mkdir(parents=True,exist_ok=True)
    suite=Path(tempfile.mkdtemp(prefix='qcml_aimd_dt25_',dir=str(parent)))
    print('suite_dir =',suite,flush=True)
    lock=read_json(PACKAGE/'source_lock.json')
    plan={'version':lock['version'],'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
          'source_base':str(base),'source_run':str(runtime),'candidate_checks':candidates,
          'python_requested':python,'dt_old':5.,'baseline_replay_frames':3,'dt_new':2.5,
          'new_frames':11385,'end_time_au':28460.,'force_verification':'USER_CONFIRMED_COMPLETED_NOT_REPEATED',
          'preserve_legacy_model_and_input_context':True,'paper_identity_not_independently_certified':True,
          'no_environment_install_or_upgrades':True,
          'package_hashes':{p.name:sha(p) for p in [PACKAGE/'compare_dt.py',PACKAGE/'observe_md.py',PACKAGE/'source_lock.json']}}
    write_json(suite/'plan.json',plan)
    state={'status':'RUNNING','stage':'snapshot','new_md_completed':False}
    write_json(suite/'status.json',state)
    try:
        baseline_path=snapshot(base,runtime,suite)
        baseline=parse_log(baseline_path);check_grid(baseline,5.,5693)
        write_table(suite/'baseline_dt5.tsv',baseline)
        source_prefix=parse_log(PACKAGE/'evidence/old_long_prefix.log')
        check(prefix_comparison(baseline,source_prefix)['passed'],'Frozen baseline prefix mismatch')
        replay=make_case(suite,'replay_dt5',5.,3)
        state['stage']='environment';write_json(suite/'status.json',state)
        check_environment(replay,python,suite)
        state['stage']='three_record_runtime_replay';write_json(suite/'status.json',state)
        execute([python,'-B',str(PACKAGE/'observe_md.py'),str(replay/'md.py')],replay,replay/'md.log',replay/'md.err')
        rr=parse_log(replay/'md.log');check_grid(rr,5.,3)
        comparison=prefix_comparison(rr,baseline);write_json(suite/'runtime_replay.json',comparison)
        check(comparison['passed'],'Runtime no longer reproduces original prefix; dt2.5 was NOT started')
        verify_inputs(suite)
        half=make_case(suite,'dt2p5',2.5,11385)
        state['stage']='dt2p5_production';write_json(suite/'status.json',state)
        # The observation wrapper checks time zero against the already-replayed initial state.
        execute([python,'-B',str(PACKAGE/'observe_md.py'),str(half/'md.py'),
                 '--initial-observation',str(replay/'observations.jsonl')],half,half/'md.log',half/'md.err')
        rows=parse_log(half/'md.log');check_grid(rows,2.5,11385)
        initial=prefix_comparison(rows,baseline,count=1);write_json(suite/'initial_state_comparison.json',initial)
        check(initial['passed'],'Initial state differs')
        write_table(suite/'dt2p5.tsv',rows)
        stats={'baseline_dt5':energy_statistics(baseline),'dt2p5_all':energy_statistics(rows),
               'dt2p5_at_old_times':energy_statistics(rows[::2]),'same_physical_window':True,
               'no_preassigned_ratio_or_convergence_pass':True}
        write_json(suite/'comparison.json',stats)
        verify_inputs(suite)
        state.update(status='COMPLETED_TIMESTEP_COMPARISON',stage='done',new_md_completed=True,
                     baseline_prefix_replayed=True)
    except Exception as exc:
        state.update(status='FAILED_OR_INCOMPLETE',error=repr(exc),traceback=traceback.format_exc())
        try:verify_inputs(suite)
        except Exception as e:state['integrity_check_error']=repr(e)
        traceback.print_exc()
    finally:
        write_json(suite/'status.json',state)
        try:collect(suite)
        except Exception as e:print('collection_error =',repr(e),flush=True)
    print('comparison_status =',state['status'],flush=True)
    return 0 if state['status']=='COMPLETED_TIMESTEP_COMPARISON' else 2


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    r=sub.add_parser('run');r.add_argument('--base',required=True);r.add_argument('--run-dir')
    r.add_argument('--out-parent',required=True);r.add_argument('--python',default=sys.executable)
    c=sub.add_parser('collect');c.add_argument('--suite',required=True)
    x=p.parse_args()
    if x.cmd=='collect':collect(x.suite);return 0
    for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,interrupt)
    return run(x.base,x.run_dir,x.out_parent,x.python)


if __name__=='__main__':
    try:sys.exit(main())
    except Exception:traceback.print_exc();sys.exit(2)
