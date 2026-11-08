"""Exact input snapshots and minimal force/handoff edits in a new case."""
import ast
import difflib
import importlib.util
import shutil
from pathlib import Path
from common import require,load,dump,sha,safe_file,safe_parent,MIB

PACKAGE=Path(__file__).resolve().parent

def vendor():
    spec=importlib.util.spec_from_file_location('frozen_legacy_dt',str(PACKAGE/'vendor/compare_dt.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def check_parent(parent):
    parent=Path(parent).expanduser().resolve();lock=load(PACKAGE/'parent_lock.json')
    require(parent.is_dir(),'Missing full parent MD suite')
    for x in lock['files']:
        p=safe_file(parent,x['relative_path'])
        require(p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'Parent input mismatch: '+str(p))
    p=safe_file(parent,'dt2p5/observations.jsonl')
    require(p.stat().st_size<128*MIB,'Unexpected observation file size')
    return {'parent':str(parent),'locked_files_checked':len(lock['files']),
            'observations_bytes':p.stat().st_size,'observations_sha256':sha(p),
            'observations_hash_previously_uploaded':False,
            'observations_contents_will_be_cross_checked_against_frozen_dt2p5_table':True}


def patch_integrator(text):
    old="""        grad_new = np.loadtxt('force_true.txt')
        grad[0,-1] = grad_new[2]
        grad[1,-1] = -grad_new[2]"""
    new="""        grad_new = np.loadtxt('force_true.txt')
        from state_bridge import consume
        e_tot, grad = consume(self.mol, e_tot, grad_new)"""
    require(text.count(old)==1,'Original handoff block not uniquely identified')
    revised=text.replace(old,new)
    require(revised.count(new)==1 and revised.replace(new,old)==text,'Handoff change not reversible')
    # No changes to Verlet update, masses, time advancement, or output writer.
    a,b=ast.parse(text),ast.parse(revised)
    for tree in [a,b]:
        c=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='VelocityVerlet')
        f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_compute_accel');f.body=[ast.Pass()]
    require(ast.dump(a)==ast.dump(b),'Change outside acceleration handoff')
    return revised


def stage(parent,suite,protocol):
    parent=Path(parent).resolve();suite=Path(suite).resolve()
    report=check_parent(parent);dump(suite/'parent_preflight.json',report)
    manifest=[]
    for x in load(PACKAGE/'parent_lock.json')['files']:
        rel=x['relative_path']
        if not (rel.startswith('original_code/') or rel.startswith('inputs/')):continue
        src=safe_file(parent,rel);dst=suite/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
        require(sha(dst)==x['sha256'],'Copy mismatch')
        dst.chmod(0o444)
        manifest.append({'source':str(src),'copy':str(dst),'sha256':x['sha256'],'bytes':x['bytes']})
    dump(suite/'input_manifest.json',manifest)
    case=vendor().make_case(suite,'corrected_dt2p5',protocol['dt_au'],protocol['frames'])
    original=case/'force.py';candidate=PACKAGE/'candidate/force_overlap_symmetry_only.py'
    require(sha(original)==protocol['force_old_sha256'],'Wrong original force')
    require(sha(candidate)==protocol['force_corrected_sha256'],'Wrong candidate force')
    before=original.read_bytes();after=candidate.read_bytes()
    require(before.count(b'return (s_r + s_r).transpose((0, 2, 1))')==1,'Missing exact overlap expression')
    require(before.replace(b'return (s_r + s_r).transpose((0, 2, 1))',b'return s_r + s_r.transpose((0, 2, 1))')==after,'More than the one overlap edit')
    original.write_bytes(after)
    p=case/'integrators.py';b=p.read_text();a=patch_integrator(b);p.write_text(a)
    diffs=''.join(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),fromfile='original/force.py',tofile='runtime/force.py'))
    diffs+=''.join(difflib.unified_diff(b.splitlines(True),a.splitlines(True),fromfile='original/integrators.py',tofile='runtime/integrators.py'))
    (suite/'scientific_changes.diff').write_text(diffs)
    for name in ['state_bridge.py','md_observer.py','common.py']:shutil.copyfile(PACKAGE/name,case/name)
    # make_case's manifest describes its relocation only; save actual final identities separately.
    keys=[p.name for p in case.glob('*.py')]+['transformer_model_params.pth','zeroindex_LiH.txt']
    runtime={name:sha(case/name) for name in sorted(keys)}
    dump(case/'final_runtime_lock.json',runtime)
    dump(case/'patch_provenance.json',{'pre_patch_relocation_manifest':'runtime_changes.json',
        'actual_final_hashes':'final_runtime_lock.json','force_overlap_single_line':True,
        'all_six_returned_gradients_used':True,'predicted_energy_returned_as_MD_potential':True,
        'Verlet_formula_changed':False,'initial_velocities_changed':False,'parameter_optimization_added':False,
        'reference_scanner_is_diagnostic_only':True})
    for name in keys:(case/name).chmod(0o444)
    return case


def verify_inputs(suite):
    suite=Path(suite);rows=[]
    for x in load(suite/'input_manifest.json'):
        rows.append({'source':x['source'],'copy':x['copy'],
            'source_unchanged':Path(x['source']).is_file() and sha(x['source'])==x['sha256'],
            'copy_unchanged':Path(x['copy']).is_file() and sha(x['copy'])==x['sha256']})
    r=load(suite/'parent_preflight.json');parent=Path(r['parent'])
    rows.append({'source':str(parent/'dt2p5/observations.jsonl'),'copy':None,
       'source_unchanged':sha(parent/'dt2p5/observations.jsonl')==r['observations_sha256'],'copy_unchanged':True})
    for x in load(PACKAGE/'parent_lock.json')['files']:
        if x['relative_path'].startswith(('original_code/','inputs/')):continue
        p=parent/x['relative_path'];rows.append({'source':str(p),'copy':None,'source_unchanged':sha(p)==x['sha256'],'copy_unchanged':True})
    dump(suite/'input_integrity_after.json',rows)
    require(all(x['source_unchanged'] and x['copy_unchanged'] for x in rows),'Protected input changed')
    return rows
