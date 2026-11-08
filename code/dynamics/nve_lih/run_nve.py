#!/usr/bin/env python3
"""New corrected-gradient NVE only; old dt=2.5 trajectory is read, never rerun."""
import argparse
import datetime
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import traceback
from common import require,load,dump,sha,safe_parent,verify_package,read_observations,scalar_rows,statistics,ledger,write_table,read_table,MIB
import staging

PACKAGE=Path(__file__).resolve().parent
_child=None

def execute(cmd,cwd,stdout,stderr):
    global _child
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='')
    with Path(stdout).open('w') as a,Path(stderr).open('w') as b:
        _child=subprocess.Popen(cmd,cwd=str(cwd),stdout=a,stderr=b,env=env)
        try:rc=_child.wait()
        finally:_child=None
    require(rc==0,'Child failed (%d): %s; %s'%(rc,stdout,stderr))

def interrupt(signum,frame):
    if _child is not None:
        _child.terminate()
        try:_child.wait(timeout=25)
        except subprocess.TimeoutExpired:_child.kill();_child.wait()
    raise RuntimeError('Signal '+str(signum)+'; no restart, completed frames retained')

def baseline(parent,suite,protocol):
    # The historical dt=2.5 trajectory has 11,385 frames; the new short run
    # deliberately has fewer. Validate the entire historical record using its
    # own frozen count, rather than incorrectly treating it as incomplete.
    saved=read_table(Path(parent)/'dt2p5.tsv')
    original_frames=int(protocol['old_baseline_frames'])
    require(len(saved)==original_frames and original_frames>=protocol['frames'],
            'Historical full dt2p5 reference length changed or shorter than planned run')
    require(abs(float(saved[-1]['time_au'])-float(protocol['old_baseline_end_time_au']))<1e-9,
            'Historical baseline end time changed')
    old_protocol=dict(protocol);old_protocol['frames']=original_frames
    obs,_=read_observations(Path(parent)/'dt2p5/observations.jsonl',old_protocol)
    rows=scalar_rows(obs,old_protocol)
    require(len(saved)==len(rows),'Old table count differs')
    maxima={k:0. for k in ['R_angstrom','E_pred_Eh','Ekin_Eh','Etot_pred_Eh','force_scalar']}
    for a,b,o in zip(rows,saved,obs):
        require(float(a['time_au'])==float(b['time_au']),'Old time mismatch')
        for key,tol in [('R_angstrom',1e-9),('E_pred_Eh',1e-9),('Ekin_Eh',1e-10),('Etot_pred_Eh',1e-9)]:
            d=abs(float(a[key])-float(b[key]));maxima[key]=max(maxima[key],d);require(d<tol,'Old observations mismatch saved '+key)
        d=abs(a['gLi_z']-float(b['force_scalar']));maxima['force_scalar']=max(maxima['force_scalar'],d)
        require(d<1e-8,'Old applied Li-z does not match frozen table')
    dump(suite/'initial_reference.json',obs[0]);write_table(suite/'old_dt2p5.tsv',rows)
    stats=statistics(rows);check_ledger,_=ledger(obs,protocol)
    dump(suite/'old_baseline_check.json',{'passed':True,'max_differences_vs_locked_printed_table':maxima,
        'statistics_high_precision':stats,'ledger':check_ledger,
        'applied_old_gradient_is_not_new_corrected_gradient':True})
    return rows

def analyze(suite,allow_partial=False,make_plots=True):
    suite=Path(suite).resolve();plan=load(suite/'plan.json');protocol=plan['protocol']
    path=suite/'corrected_dt2p5/observations.jsonl'
    require(path.is_file(),'No new completed observations yet')
    obs,issue=read_observations(path,protocol,True,allow_partial)
    require(obs,'No valid completed frame')
    rows=scalar_rows(obs,protocol);write_table(suite/'corrected_dt2p5.tsv',rows)
    complete=(len(rows)==protocol['frames'] and issue is None)
    old=read_table(suite/'old_dt2p5.tsv')
    comparison={'new_frames':len(rows),'planned_frames':protocol['frames'],'new_trajectory_complete':complete,
      'partial_issue':issue,'definition':'E_pred+K at same frame; no smoothing/detrending/phase alignment',
      'initial_reference':'same old q/v/m; corrected gradient expected to differ',
      'energy_conservation_pass':None,'convergence_claim':False,'VQE_optimization_performed':False}
    if len(rows)>=2:
        aligned=old[:len(rows)]
        require(len(aligned)==len(rows) and all(float(a['time_au'])==float(b['time_au']) for a,b in zip(aligned,rows)),'Unequal time grids')
        os_=statistics(aligned);ns=statistics(rows)
        comparison.update(old_same_window=os_,new_same_window=ns,same_physical_window=True,
            ratios_new_over_old={k:(ns[k]/os_[k] if os_[k]!=0 else None) for k in ['max_abs_delta_Eh','rms_from_initial_Eh','peak_to_peak_Eh']})
        pairs=[{'time_au':n['time_au'],'time_fs':n['time_fs'],'old_delta_Etot_Eh':float(o['delta_Etot_Eh']),
                'new_delta_Etot_Eh':n['delta_Etot_Eh'],'new_minus_old_R_A':n['R_angstrom']-float(o['R_angstrom'])} for o,n in zip(aligned,rows)]
        write_table(suite/'equal_time_comparison.tsv',pairs)
    ls,lr=ledger(obs,protocol);dump(suite/'new_ledger_summary.json',ls);write_table(suite/'new_step_ledger.tsv',lr)
    dump(suite/'comparison.json',comparison)
    if make_plots and len(rows)>=2:
        try:
            import plot_results
            plot_results.make(suite)
            dump(suite/'plot_status.json',{'status':'CREATED','scientific_results_changed':False})
        except Exception as exc:
            dump(suite/'plot_status.json',{'status':'NOT_CREATED','error':repr(exc),'scientific_results_unchanged':True,
                'manual_command':'python plot_results.py --suite '+str(suite)})
    return comparison

def collect(suite):
    suite=Path(suite).expanduser().resolve()
    require(load(suite/'plan.json')['version']=='corrected_force_nve_short_v1','Wrong suite kind')
    archive=suite.parent/(suite.name+'_audit.tar.gz')
    require(not archive.exists(),'Audit already exists; refusing overwrite')
    # An uncatchable job kill can leave observations but no summary. Rebuild read-only results here.
    if (suite/'corrected_dt2p5/observations.jsonl').is_file() and not (suite/'comparison.json').exists():
        try:analyze(suite,allow_partial=True,make_plots=False)
        except Exception as exc:dump(suite/'partial_analysis_error.json',{'error':repr(exc)})
    staging_dir=Path(tempfile.mkdtemp(prefix='nve_audit_text_',dir=str(suite.parent)))
    out=staging_dir/suite.name;out.mkdir();manifest=[];total=0
    selected=list(suite.glob('*.json'))+list(suite.glob('*.tsv'))+list(suite.glob('*.diff'))
    selected+=list(suite.glob('corrected_dt2p5/*.json'))+list(suite.glob('code/**/*.py'))+list(suite.glob('code/**/*.json'))
    for p in sorted(set(selected)):
        require(not p.is_symlink(),'Unexpected output symlink')
        size=p.stat().st_size;require(size<=8*MIB and total+size<=24*MIB,'Text collection budget exceeded')
        q=out/p.relative_to(suite);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());total+=size
        manifest.append({'relative_path':str(q.relative_to(out)),'sha256':sha(q),'bytes':size,'status':'INCLUDED'})
    for p in sorted(list(suite.glob('*.log'))+list(suite.glob('*.err'))+list(suite.glob('corrected_dt2p5/*.log'))+list(suite.glob('corrected_dt2p5/*.err'))):
        require(not p.is_symlink(),'Unexpected log symlink')
        with p.open('rb') as f:
            if p.stat().st_size<=48000: content=f.read();excerpt=False
            else:
                head=f.read(16000);f.seek(-32000,2);tail=f.read();excerpt=True
                content=head+b'\n[EXPLICIT HEAD/TAIL EXCERPT; MIDDLE OMITTED]\n'+tail
        q=out/'logs'/str(p.relative_to(suite)).replace('/','__');q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(content)
        manifest.append({'relative_path':str(q.relative_to(out)),'sha256':sha(q),'bytes':len(content),'status':'EXCERPT' if excerpt else 'INCLUDED',
                         'original_bytes':p.stat().st_size})
    for label in ['inputs/','original_code/','corrected_dt2p5/observations.jsonl','corrected_dt2p5/*.xyz','figures/']:
        manifest.append({'relative_path':label,'status':'INTENTIONALLY_EXCLUDED','reason':'large raw arrays/model/duplicate code or reproducible plots remain on server'})
    dump(out/'archive_manifest.json',manifest)
    partial=archive.with_name(archive.name+'.partial')
    with partial.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as t:t.add(out,arcname=out.name)
    require(partial.stat().st_size<=4*MIB,'Audit exceeds 4 MiB: partial archive NOT published; scientific files remain intact')
    partial.rename(archive);shutil.rmtree(staging_dir)
    print('audit_tarball =',archive,flush=True);print('audit_KiB =',round(archive.stat().st_size/1024,3),flush=True)
    return archive

def copy_code(suite):
    manifest=load(PACKAGE/'PACKAGE_SHA256.json')
    for rel in manifest:
        if rel.startswith(('tests/','validation/','evidence/')):continue
        p=PACKAGE/rel;q=suite/'code'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
    dump(suite/'code/PACKAGE_SHA256_record.json',manifest)

def run(parent,out_parent,python):
    verify_package(PACKAGE);parent=Path(parent).expanduser().resolve()
    out_parent=safe_parent(out_parent,[parent,PACKAGE]);out_parent.mkdir(parents=True,exist_ok=True)
    preflight=staging.check_parent(parent)
    suite=Path(tempfile.mkdtemp(prefix='qcml_aimd_corrected_nve_',dir=str(out_parent)))
    print('suite_dir =',suite,flush=True)
    protocol=load(PACKAGE/'protocol.json');plan={'version':protocol['version'],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'parent_md_suite':str(parent),'python_requested':str(python),'protocol':protocol,'package_hashes':load(PACKAGE/'PACKAGE_SHA256.json'),
        'purpose':'test current directly predicted-state corrected-force NVE; no added VQE optimization'}
    dump(suite/'plan.json',plan);dump(suite/'protocol.json',protocol);copy_code(suite)
    state={'status':'RUNNING','stage':'snapshot','new_MD_completed':False,'energy_conservation_pass':None}
    dump(suite/'status.json',state)
    try:
        case=staging.stage(parent,suite,protocol)
        state['stage']='read_existing_dt2p5';dump(suite/'status.json',state);baseline(parent,suite,protocol)
        state['stage']='runtime_imports';dump(suite/'status.json',state)
        # Reuse exact old import checker; it does not perform molecular calculations.
        m=staging.vendor();m.execute=execute;m.check_environment(case,python,suite)
        state['stage']='corrected_force_NVE';dump(suite/'status.json',state)
        execute([python,'-B',str(case/'md_observer.py'),'--driver',str(case/'md.py'),
            '--initial',str(suite/'initial_reference.json'),'--protocol',str(suite/'protocol.json'),
            '--runtime-lock',str(case/'final_runtime_lock.json')],case,case/'md.log',case/'md.err')
        state['stage']='analyze_equal_time_window';dump(suite/'status.json',state)
        comparison=analyze(suite)
        require(comparison['new_trajectory_complete'],'New trajectory incomplete')
        require(load(case/'initial_check.json')['passed'],'Initial check missing or failed')
        require(all(load(case/'provenance.json')['runtime_hashes_unchanged'].values()),'Runtime integrity failed')
        staging.verify_inputs(suite)
        state.update(status='COMPLETED_CORRECTED_FORCE_NVE',stage='done',new_MD_completed=True,
                     frames=protocol['frames'],old_MD_rerun=False,energy_conservation_pass=None)
    except BaseException as exc:
        state.update(status='FAILED_OR_INCOMPLETE',error=repr(exc),traceback=traceback.format_exc());traceback.print_exc()
        try:
            if (suite/'corrected_dt2p5/observations.jsonl').exists():analyze(suite,True,False)
        except Exception as e:state['partial_analysis_error']=repr(e)
        try:
            if (suite/'input_manifest.json').exists():staging.verify_inputs(suite)
        except Exception as e:state['integrity_error']=repr(e)
    finally:
        dump(suite/'status.json',state)
        try:collect(suite)
        except Exception as e:print('collection_error =',repr(e),flush=True)
    print('run_status =',state['status'],flush=True)
    return 0 if state['status']=='COMPLETED_CORRECTED_FORCE_NVE' else 2

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    r=sub.add_parser('run');r.add_argument('--parent',required=True);r.add_argument('--out-parent',required=True);r.add_argument('--python',default=sys.executable)
    for name in ['analyze','collect']:
        q=sub.add_parser(name);q.add_argument('--suite',required=True)
    x=p.parse_args()
    if x.cmd=='collect':collect(x.suite);return 0
    if x.cmd=='analyze':analyze(x.suite,True);return 0
    for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,interrupt)
    return run(x.parent,x.out_parent,x.python)

if __name__=='__main__':
    try:sys.exit(main())
    except Exception:traceback.print_exc();sys.exit(2)
