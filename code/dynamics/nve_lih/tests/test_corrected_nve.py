import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

import common as c
import staging
import state_bridge as b
import md_observer
import run_nve
import reuse_interrupted_prefix as rip
import plot_results

PKG=Path(__file__).resolve().parents[1]

def protocol():return c.load(PKG/'protocol.json')

class FakeMol:
    def __init__(self,q=None,names=None):
        self.q=np.array(q if q is not None else [[0,0,0],[0,0,1.5475/.52917721092]],float)
        self.names=names or ['Li','H']
    def atom_coords(self,*args,**kwargs):return self.q*(.52917721092 if kwargs.get('unit')=='ANG' else 1.)
    def atom_symbol(self,i):return self.names[i]

class Bridge(unittest.TestCase):
    def setUp(self):b.configure(protocol());self.t=np.zeros(330);self.g=np.array([[0,0,.00032],[0,0,-.00032]])
    def ready(self):b.record_prediction(-7.88,self.t);b.record_gradient(self.g,self.t)
    def test_full_components_unchanged(self):
        self.g[0,0]=1e-10;self.g[1,2]+=2e-9;self.ready()
        e,g=b.consume(FakeMol(),-7.89,self.g.ravel());np.testing.assert_array_equal(g,self.g);self.assertEqual(e,-7.88)
        self.assertNotEqual(g[1,2],-g[0,2])
    def test_epot_replaced_not_reference(self):
        self.ready();e,g=b.consume(FakeMol(),-7.89,self.g);self.assertEqual(e,-7.88);self.assertEqual(b.peek()['scanner_Epot_Eh'],-7.89)
    def test_disk_mismatch_rejected(self):
        self.ready()
        with self.assertRaises(ValueError):b.consume(FakeMol(),-7.89,self.g+1e-8)
    def test_wrong_atom_order(self):
        self.ready()
        with self.assertRaises(ValueError):b.consume(FakeMol(names=['H','Li']),-7.89,self.g)
    def test_nonaxial_rejected(self):
        self.ready()
        with self.assertRaises(ValueError):b.consume(FakeMol([[0,0,0],[.1,0,3]]),-7.89,self.g)
    def test_reversed_axis_rejected(self):
        self.ready()
        with self.assertRaises(ValueError):b.consume(FakeMol([[0,0,0],[0,0,-3]]),-7.89,self.g)
    def test_range_rejected(self):
        self.ready()
        with self.assertRaises(ValueError):b.consume(FakeMol([[0,0,0],[0,0,10]]),-7.89,self.g)
    def test_complex_energy_rejected(self):
        with self.assertRaises(ValueError):b.record_prediction(1+1e-3j,self.t)
    def test_small_imag_energy_checked(self):b.record_prediction(-7.88+1e-13j,self.t);self.assertEqual(b._record['E_pred_Eh'],-7.88)
    def test_nonfinite_energy(self):
        with self.assertRaises(ValueError):b.record_prediction(float('nan'),self.t)
    def test_parameter_count(self):
        with self.assertRaises(ValueError):b.record_prediction(1.,[0.]*329)
    def test_parameter_mutation(self):
        b.record_prediction(-7.88,self.t);self.t[5]=1e-5
        with self.assertRaises(ValueError):b.record_gradient(self.g,self.t)
    def test_complex_gradient(self):
        b.record_prediction(-7.88,self.t)
        with self.assertRaises(ValueError):b.record_gradient(self.g+1e-3j,self.t)
    def test_nonfinite_gradient(self):
        b.record_prediction(-7.88,self.t);self.g[0,0]=np.nan
        with self.assertRaises(ValueError):b.record_gradient(self.g,self.t)
    def test_net_gradient_is_guard_not_projection(self):
        b.record_prediction(-7.88,self.t);self.g[1,2]=.001
        with self.assertRaises(ValueError):b.record_gradient(self.g,self.t)
    def test_gradient_without_prediction(self):
        with self.assertRaises(ValueError):b.record_gradient(self.g,self.t)
    def test_double_prediction(self):
        b.record_prediction(-7.88,self.t)
        with self.assertRaises(ValueError):b.record_prediction(-7.88,self.t)
    def test_double_consume(self):
        self.ready();b.consume(FakeMol(),-7.89,self.g)
        with self.assertRaises(ValueError):b.consume(FakeMol(),-7.89,self.g)
    def test_finish_and_new_frame(self):
        self.ready();b.consume(FakeMol(),-7.89,self.g);b.finish_frame();self.ready();self.assertEqual(b.completed_frames(),1)
    def test_gradient_result_not_mutated(self):
        self.ready();e,g=b.consume(FakeMol(),-7.89,self.g);g[0,0]=2;self.assertEqual(b.peek()['returned_gradient_Eh_per_bohr'][0][0],0)


def simulate(n=21,dt=.02,force_constant=.01):
    """Synthetic two-mass harmonic trajectory. Not a LiH calculation."""
    q=np.array([[0.,0.,0.],[0.,0.,1.]])
    v=np.array([[0.,0.,.002],[0.,0.,-.003]])
    m=np.array([2.,3.]);rows=[];prior=None
    for i in range(n):
        if i:q=q+dt*v-.5*dt**2*prior/m[:,None]
        r=q[1,2]-q[0,2];s=force_constant*(r-.8);g=np.array([[0,0,-s],[0,0,s]],float)
        if i:v=v-.5*dt*(prior+g)/m[:,None]
        energy=-1.+.5*.01*(r-.8)**2;k=c.kinetic(m,v)
        row={'time_au':i*dt,'dt_au':dt,'E_pred_Eh':energy,'Ekin_Eh':k,'Etot_pred_Eh':energy+k,
             'scanner_Epot_Eh':energy-.1,'MD_Epot_Eh':energy,'coordinates_bohr':q.tolist(),
             'velocities_au':v.tolist(),'masses_au':m.tolist(),'applied_gradient_Eh_per_bohr':g.tolist(),
             'returned_gradient_Eh_per_bohr':g.tolist(),'parameter_sha256_float64':'SYNTHETIC'}
        rows.append(row);prior=g.copy()
    return rows

def tiny_protocol():
    p=protocol();p.update(dt_au=.02,frames=21,end_time_au=.4,old_baseline_frames=21,old_baseline_end_time_au=.4,initial_prediction_Eh=-.9998,
       initial_corrected_gradient_Eh_per_bohr=[[0,0,-.002],[0,0,.002]])
    return p

class Numerics(unittest.TestCase):
    def test_verlet_synthetic(self):
        summary,rows=c.ledger(simulate(),tiny_protocol())
        self.assertLess(summary['maxima']['q_update_bohr'],1e-15)
        self.assertLess(summary['maxima']['v_update_au'],1e-16)
    def test_bad_verlet_detected_in_ledger(self):
        rows=simulate();rows[7]['coordinates_bohr'][0][2]+=.01
        summary,_=c.ledger(rows,tiny_protocol());self.assertGreater(summary['maxima']['q_update_bohr'],.009)
    def test_no_detrending(self):
        r=[{'time_au':i,'Etot_pred_Eh':1.+i*.001} for i in range(5)]
        s=c.statistics(r);self.assertAlmostEqual(s['max_abs_delta_Eh'],.004);self.assertAlmostEqual(s['linear_slope_Eh_per_au'],.001)
        self.assertFalse(s['detrended']);self.assertIsNone(s['conservation_pass'])
    def test_drift_units(self):
        s=c.statistics([{'time_au':0,'Etot_pred_Eh':1},{'time_au':1,'Etot_pred_Eh':1.00001}],1.)
        self.assertAlmostEqual(s['linear_slope_mHa_per_ps'],10.)
    def test_nonmonotone_time_rejected(self):
        with self.assertRaises(ValueError):c.statistics([{'time_au':1,'Etot_pred_Eh':1},{'time_au':1,'Etot_pred_Eh':1}])
    def test_observation_shapes(self):
        r=simulate()[0];r['coordinates_bohr']=[[0,0]]
        with self.assertRaises(ValueError):c.validate_observation(r,0,tiny_protocol(),True)
    def test_bad_mass_rejected(self):
        r=simulate()[0];r['masses_au'][0]=-1.
        with self.assertRaises(ValueError):c.validate_observation(r,0,tiny_protocol(),True)
    def test_old_reference_epot_not_accepted_as_new(self):
        r=simulate()[0];r['MD_Epot_Eh']=r['scanner_Epot_Eh']
        with self.assertRaises(ValueError):c.validate_observation(r,0,tiny_protocol(),True)
    def test_bad_handoff_rejected(self):
        r=simulate()[0];r['returned_gradient_Eh_per_bohr'][0][2]+=.001
        with self.assertRaises(ValueError):c.validate_observation(r,0,tiny_protocol(),True)
    def test_all_frames_no_selection(self):
        r=simulate();s=c.scalar_rows(r,tiny_protocol());self.assertEqual(len(s),21);self.assertEqual(s[0]['delta_Etot_Eh'],0)
    def test_energy_decomposition(self):
        obs=simulate(force_constant=.02);summary,rows=c.ledger(obs,tiny_protocol())
        for o,r in zip(obs,rows):self.assertAlmostEqual(r['delta_K_minus_work_Eh']+r['delta_Epred_plus_work_Eh'],o['Etot_pred_Eh']-obs[0]['Etot_pred_Eh'],places=14)
    def test_initial_gradient_allowed_to_change(self):
        old,new=simulate(force_constant=.02)[0],simulate()[0]
        check=md_observer.check_initial(new,old,tiny_protocol());self.assertTrue(check['passed'])
    def test_bad_initial_velocity(self):
        old,new=simulate(force_constant=.02)[0],simulate()[0];new['velocities_au'][0][2]+=.01
        self.assertFalse(md_observer.check_initial(new,old,tiny_protocol())['passed'])
    def test_bad_initial_candidate_gradient(self):
        old,new=simulate()[0],simulate()[0];new['returned_gradient_Eh_per_bohr'][0][2]+=.01
        self.assertFalse(md_observer.check_initial(new,old,tiny_protocol())['passed'])

class Files(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
    def tearDown(self):self.tmp.cleanup()
    def test_safe_sibling(self):
        (self.root/'a').mkdir();self.assertEqual(c.safe_parent(self.root/'ab',[self.root/'a']),self.root/'ab')
    def test_unsafe_descendant(self):
        with self.assertRaises(ValueError):c.safe_parent(self.root/'a/out',[self.root/'a'])
    def test_unsafe_symlink_parent(self):
        (self.root/'a').mkdir();(self.root/'alias').symlink_to(self.root/'a',target_is_directory=True)
        with self.assertRaises(ValueError):c.safe_parent(self.root/'alias/out',[self.root/'a'])
    def test_no_symlink_inputs(self):
        (self.root/'a').write_text('x');(self.root/'link').symlink_to(self.root/'a')
        with self.assertRaises(ValueError):c.safe_file(self.root,'link')
    def test_path_traversal(self):
        with self.assertRaises(ValueError):c.safe_file(self.root,'../other')
    def test_json_nan_rejected(self):
        with self.assertRaises(ValueError):c.dump(self.root/'bad.json',{'bad':float('nan')})
    def test_json_native_int(self):
        c.dump(self.root/'ok.json',{'count':int(np.int64(6))});self.assertEqual(c.load(self.root/'ok.json')['count'],6)
    def test_json_numpy_bool_and_int(self):
        c.dump(self.root/'native.json',{'ok':np.bool_(True),'n':np.int64(6)})
        self.assertEqual(c.load(self.root/'native.json'),{'ok':True,'n':6})
    def test_json_numpy_nan_rejected(self):
        with self.assertRaises(ValueError):c.dump(self.root/'bad.json',{'n':np.float32(np.nan)})
    def test_partial_tail(self):
        p=self.root/'o.jsonl';p.write_text(json.dumps(simulate()[0])+'\n{')
        rows,issue=c.read_observations(p,tiny_protocol(),True,True);self.assertEqual(len(rows),1);self.assertIsNotNone(issue)
    def test_wrong_count_rejected(self):
        p=self.root/'o.jsonl';p.write_text(json.dumps(simulate()[0])+'\n')
        with self.assertRaises(ValueError):c.read_observations(p,tiny_protocol(),True)
    def test_package_tamper(self):
        (self.root/'x.py').write_text('x');c.dump(self.root/'PACKAGE_SHA256.json',{'x.py':c.sha(self.root/'x.py')})
        self.assertTrue(c.verify_package(self.root)['passed']);(self.root/'x.py').write_text('y')
        with self.assertRaises(ValueError):c.verify_package(self.root)

class ActualSource(unittest.TestCase):
    def test_short_protocol_is_exactly_8000_au(self):
        p=protocol()
        self.assertEqual(p['version'],'corrected_force_nve_short_v1')
        self.assertEqual(p['frames'],3201)
        self.assertEqual(p['old_baseline_frames'],11385)
        self.assertEqual((p['frames']-1)*p['dt_au'],p['end_time_au'])
        self.assertEqual((p['old_baseline_frames']-1)*p['dt_au'],p['old_baseline_end_time_au'])

    def test_force_single_line_actual(self):
        old=(PKG/'evidence/original_code/force.py').read_bytes();new=(PKG/'candidate/force_overlap_symmetry_only.py').read_bytes()
        self.assertEqual(old.replace(b'return (s_r + s_r).transpose((0, 2, 1))',b'return s_r + s_r.transpose((0, 2, 1))'),new)
        self.assertEqual(c.sha(PKG/'candidate/force_overlap_symmetry_only.py'),protocol()['force_corrected_sha256'])
    def test_actual_integrator_patch(self):
        text=(PKG/'evidence/original_code/integrators.py').read_text();new=staging.patch_integrator(text)
        self.assertNotIn('grad[1,-1] = -grad_new[2]',new);self.assertIn('consume(self.mol, e_tot, grad_new)',new)
    def test_patch_reject_already_changed(self):
        text=(PKG/'evidence/original_code/integrators.py').read_text()
        with self.assertRaises(ValueError):staging.patch_integrator(staging.patch_integrator(text))
    def test_driver_changes_only_dt_frames(self):
        text=(PKG/'evidence/original_code/md.py').read_text();new=staging.vendor().change_md(text,2.5,protocol()['frames'])
        self.assertIn('dt=2.5',new);self.assertIn('steps=3201',new);self.assertIn('T = 300',new)
    def test_all_22_source_hashes(self):
        lock=c.load(PKG/'vendor/source_lock.json');self.assertEqual(len(lock['scientific_sources']),22)
        for r in lock['scientific_sources']:self.assertEqual(c.sha(PKG/'evidence/original_code'/r['name']),r['sha256'])
    def test_actual_acceleration_method_no_H_replacement(self):
        text=staging.patch_integrator((PKG/'evidence/original_code/integrators.py').read_text())
        tree=ast.parse(text);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VelocityVerlet')
        f=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_compute_accel')
        module=ast.Module(body=[f],type_ignores=[]);ns={'np':np};exec(compile(ast.fix_missing_locations(module),'actual_patched_integrator','exec'),ns)
        g=np.array([[0,0,.00032],[0,0,-.00032+1e-9]])
        with tempfile.TemporaryDirectory() as tmp:
            old=Path.cwd();os.chdir(tmp)
            try:
                b.configure(protocol());t=np.zeros(330)
                def fake_force(_):
                    b.record_prediction(-7.88,t);b.record_gradient(g,t);np.savetxt('force_true.txt',g.ravel())
                with patch.dict(sys.modules,{'nn_input':SimpleNamespace(input_f=lambda r:None),
                     'nn_output':SimpleNamespace(output_f=lambda name:None),'atom_force':SimpleNamespace(force_f=fake_force)}):
                    class Scanner:
                        converged=True
                        def __call__(self,mol):return -7.89,np.zeros((2,3))
                    s=SimpleNamespace(mol=FakeMol(),scanner=Scanner(),_masses=np.array([2.,3.]))
                    e,acc=ns['_compute_accel'](s);self.assertEqual(e,-7.88);np.testing.assert_allclose(acc,-g/np.array([2.,3.])[:,None],rtol=0,atol=0)
                    self.assertNotEqual(acc[1,2],g[0,2]/3.)
            finally:os.chdir(old)
    def test_python38_syntax(self):
        for p in PKG.rglob('*.py'):
            if 'evidence' in p.parts:continue
            ast.parse(p.read_text(),feature_version=(3,8))
    def test_shell_syntax(self):
        for name in ['submit_slurm_queue.sh','run_nve.slurm']:
            subprocess.run(['bash','-n',str(PKG/name)],check=True)
    def test_launcher_does_not_count_queue(self):
        text=(PKG/'submit_slurm_queue.sh').read_text();self.assertNotIn('squeue --',text);self.assertIn('sbatch --parsable',text)
        self.assertIn('submitted_jobid.txt',text)


class Controller(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.pkg=self.root/'package';self.parent=self.root/'old';self.out=self.root/'runs'
        self.pkg.mkdir();self.parent.mkdir();self.p=tiny_protocol()
        for sub in ['vendor','candidate','evidence/original_code']:
            shutil.copytree(PKG/sub,self.pkg/sub)
        for name in ['common.py','state_bridge.py','md_observer.py','staging.py','run_nve.py','plot_results.py','protocol.json']:
            shutil.copyfile(PKG/name,self.pkg/name)
        c.dump(self.pkg/'protocol.json',self.p)
        entries=[]
        for src in (self.pkg/'evidence/original_code').glob('*.py'):
            dst=self.parent/'original_code'/src.name;dst.parent.mkdir(exist_ok=True);shutil.copyfile(src,dst)
        inp=self.parent/'inputs';inp.mkdir()
        (inp/'transformer_model_params.pth').write_text('SYNTHETIC MODEL PLACEHOLDER; NEVER LOADED')
        (inp/'zeroindex_LiH.txt').write_text('0\n');(inp/'mo').mkdir();(inp/'mo/mo_1.55.txt').write_text('SYNTHETIC')
        for rel in c.load(self.pkg/'vendor/source_lock.json')['table_paths']:
            f=inp/'data'/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('SYNTHETIC CONTEXT')
        oldrows=simulate(force_constant=.02);(self.parent/'dt2p5').mkdir()
        (self.parent/'dt2p5/observations.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in oldrows))
        table=[]
        for r in c.scalar_rows(oldrows,self.p):
            table.append(dict(time_au=r['time_au'],R_angstrom=r['R_angstrom'],E_pred_Eh=r['E_pred_Eh'],Ekin_Eh=r['Ekin_Eh'],Etot_pred_Eh=r['Etot_pred_Eh'],force_scalar=r['gLi_z']))
        c.write_table(self.parent/'dt2p5.tsv',table)
        for f in self.parent.rglob('*'):
            if f.is_file() and f.name!='observations.jsonl':entries.append({'relative_path':str(f.relative_to(self.parent)),'bytes':f.stat().st_size,'sha256':c.sha(f)})
        c.dump(self.pkg/'parent_lock.json',{'files':entries})
        c.dump(self.pkg/'PACKAGE_SHA256.json',{str(f.relative_to(self.pkg)):c.sha(f) for f in self.pkg.rglob('*') if f.is_file()})
        self.before={str(f):c.sha(f) for f in self.parent.rglob('*') if f.is_file()}
    def tearDown(self):self.tmp.cleanup()
    def executor(self,cmd,cwd,stdout,stderr):
        Path(stdout).write_text('SYNTHETIC EXECUTOR: no molecular computation\n');Path(stderr).write_text('')
        if '-c' in cmd:
            c.dump(Path(cmd[-1]),{'modules':{n:{'file':str(Path(cwd)/(n+'.py'))} for n in ['integrators','nn_input','nn_output','atom_force']},'ready':True,'synthetic':True});return
        case=Path(cwd);rows=simulate();baseline=c.load(Path(cmd[cmd.index('--initial')+1]));p=c.load(Path(cmd[cmd.index('--protocol')+1]))
        check=md_observer.check_initial(rows[0],baseline,p);self.assertTrue(check['passed']);c.dump(case/'initial_check.json',check)
        (case/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        c.dump(case/'provenance.json',{'runtime_hashes_unchanged':{k:True for k in c.load(case/'final_runtime_lock.json')},'synthetic':True})
    def go(self,executor=None):
        with patch.object(run_nve,'PACKAGE',self.pkg),patch.object(staging,'PACKAGE',self.pkg),patch.object(run_nve,'execute',executor or self.executor),patch.object(plot_results,'make',return_value=None):
            rc=run_nve.run(self.parent,self.out,sys.executable)
        suite=next(p for p in self.out.glob('qcml_aimd_corrected_nve_*') if p.is_dir());return rc,suite
    def test_complete_controller(self):
        rc,suite=self.go();self.assertEqual(rc,0);self.assertEqual(c.load(suite/'status.json')['status'],'COMPLETED_CORRECTED_FORCE_NVE')
        self.assertIsNone(c.load(suite/'comparison.json')['energy_conservation_pass']);self.assertEqual(len(c.read_table(suite/'corrected_dt2p5.tsv')),21)
        self.assertTrue((suite/'scientific_changes.diff').is_file())
    def test_short_reference_reads_full_old_and_restricts_comparison_window(self):
        # Original reference stores 21 synthetic records; shortened protocol plans 9.
        p=dict(self.p)
        p.update(frames=9,end_time_au=8*p['dt_au'],old_baseline_frames=21,
                 old_baseline_end_time_au=20*p['dt_au'])
        c.dump(self.pkg/'protocol.json',p)
        suite=self.out/'inspect_short';suite.mkdir(parents=True)
        source_rows=run_nve.baseline(self.parent,suite,p)
        self.assertEqual(len(source_rows),21)
        self.assertEqual(len(c.read_table(suite/'old_dt2p5.tsv')),21)
        c.dump(suite/'plan.json',{'version':'corrected_force_nve_short_v1','protocol':p})
        target=simulate(n=9)
        case=suite/'corrected_dt2p5';case.mkdir()
        (case/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in target))
        summary=run_nve.analyze(suite,allow_partial=False,make_plots=False)
        self.assertTrue(summary['new_trajectory_complete'])
        self.assertEqual(summary['new_frames'],9)
        self.assertEqual(summary['old_same_window']['records'],9)
        self.assertEqual(summary['new_same_window']['records'],9)
        self.assertEqual(len(c.read_table(suite/'equal_time_comparison.tsv')),9)

    def test_readonly_reuse_of_terminated_run_with_short_prefix(self):
        # Synthetic interrupted 21-frame history, predeclare a 9-frame prefix.
        oldproto=dict(self.p)
        oldproto.update(version='corrected_force_nve_v1',frames=21,end_time_au=20*self.p['dt_au'])
        oldsuite=self.out/'interrupted';oldsuite.mkdir(parents=True)
        c.dump(oldsuite/'plan.json',{'version':'corrected_force_nve_v1','protocol':oldproto})
        c.dump(oldsuite/'status.json',{'status':'FAILED_OR_INCOMPLETE','error':'synthetic SIGTERM'})
        c.dump(oldsuite/'initial_reference.json',simulate(n=21,force_constant=.02)[0])
        corrected=oldsuite/'corrected_dt2p5';corrected.mkdir()
        c.dump(corrected/'initial_check.json',{'passed':True})
        (corrected/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in simulate(n=12,force_constant=.02)))
        c.write_table(oldsuite/'old_dt2p5.tsv',c.scalar_rows(simulate(n=21,force_constant=.02),oldproto))
        short=dict(self.p)
        short.update(frames=9,end_time_au=8*self.p['dt_au'],old_baseline_frames=21,old_baseline_end_time_au=20*self.p['dt_au'])
        with patch.object(rip,'PACKAGE',self.pkg):
            dest=rip.reuse(oldsuite,self.out/'out',short)
        state=c.load(dest/'status.json')
        self.assertEqual(state['status'],'PREFIX_AVAILABLE')
        self.assertFalse(state['source_full_688fs_completed'])
        self.assertTrue((dest/'energy_comparison.json').is_file())
        self.assertEqual(len(c.read_table(dest/'new_prefix.tsv')),9)
        self.assertEqual(c.load(dest/'energy_comparison.json')['old_same_window']['records'],9)
        # If too few frames, do not invent a comparison.
        (corrected/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in simulate(n=6,force_constant=.02)))
        with patch.object(rip,'PACKAGE',self.pkg):
            dest2=rip.reuse(oldsuite,self.out/'out',short)
        self.assertEqual(c.load(dest2/'status.json')['status'],'INSUFFICIENT_PREFIX')
        self.assertFalse((dest2/'energy_comparison.json').exists())

    def test_parent_inputs_unchanged(self):
        rc,suite=self.go();self.assertEqual(self.before,{str(f):c.sha(f) for f in self.parent.rglob('*') if f.is_file()})
    def test_archive_no_model_or_arrays(self):
        rc,suite=self.go();p=Path(str(suite)+'_audit.tar.gz')
        with tarfile.open(p) as t:
            names=t.getnames();self.assertTrue(any(x.endswith('comparison.json') for x in names));self.assertFalse(any(x.endswith(('.pth','.npz','observations.jsonl')) for x in names))
        self.assertLess(p.stat().st_size,4*c.MIB)
    def test_missing_parent_observations_rejected(self):
        (self.parent/'dt2p5/observations.jsonl').unlink()
        with patch.object(staging,'PACKAGE',self.pkg):
            with self.assertRaises(ValueError):staging.check_parent(self.parent)
    def test_parent_tamper_rejected(self):
        (self.parent/'original_code/force.py').write_text('bad')
        with patch.object(staging,'PACKAGE',self.pkg):
            with self.assertRaises(ValueError):staging.check_parent(self.parent)
    def test_child_failure_records_partial(self):
        def fail(cmd,cwd,stdout,stderr):
            self.executor(cmd,cwd,stdout,stderr)
            if '-c' not in cmd:
                rows=simulate()[:3];(Path(cwd)/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));raise ValueError('synthetic child failure')
        rc,suite=self.go(fail);self.assertEqual(rc,2);self.assertEqual(c.load(suite/'status.json')['status'],'FAILED_OR_INCOMPLETE')
        self.assertFalse(c.load(suite/'comparison.json')['new_trajectory_complete']);self.assertEqual(c.load(suite/'comparison.json')['new_frames'],3)
    def test_collect_refuses_duplicate(self):
        rc,suite=self.go()
        with self.assertRaises(ValueError):run_nve.collect(suite)


class ObserverIntegration(unittest.TestCase):
    def test_real_observer_and_verlet_with_synthetic_electronics(self):
        # Executes md_observer.main and actual patched Verlet methods. Electronics are a declared harmonic stub.
        with tempfile.TemporaryDirectory() as td:
            case=Path(td).resolve();p=tiny_protocol();p['legacy_reference_R_range_A']=[.1,10.]
            originals=PKG/'evidence/original_code'
            for name in ['md.py','vqe_opt.py']:(case/name).write_bytes((originals/name).read_bytes())
            (case/'integrators.py').write_text(staging.patch_integrator((originals/'integrators.py').read_text()))
            (case/'force.py').write_bytes((PKG/'candidate/force_overlap_symmetry_only.py').read_bytes())
            (case/'state_bridge.py').write_bytes((PKG/'state_bridge.py').read_bytes())
            lock={x.name:c.sha(x) for x in case.glob('*.py')};c.dump(case/'lock.json',lock)
            c.dump(case/'protocol.json',p);c.dump(case/'initial.json',simulate(force_constant=.02)[0])
            tree=ast.parse((case/'integrators.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VelocityVerlet')
            methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_next','_compute_accel','_next_geometry','_next_velocity')]
            def toframe(x):return SimpleNamespace(time=x.time,ekin=x.ekin,epot=x.epot,coord=x.mol.q,veloc=x.veloc)
            ns={'np':np,'_toframe':toframe};exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),'real_Verlet_methods','exec'),ns)
            class Mole(FakeMol):
                def set_geom_(self,q,unit=None):self.q=np.array(q);return self
                def build(self):return self
            class Integrator:
                def __init__(self):
                    self.mol=Mole([[0,0,0],[0,0,1]])
                    self._masses=np.array([2.,3.]);self.veloc=np.array([[0,0,.002],[0,0,-.003]])
                    self.dt=p['dt_au'];self.time=0.;self.accel=None
                    class Scan:
                        converged=True
                        def __call__(s,m):return -.1+energy(m),np.zeros((2,3))
                    self.scanner=Scan()
                def compute_kinetic_energy(self):return c.kinetic(self._masses,self.veloc)
            for f in methods:setattr(Integrator,f.name,ns[f.name])
            md=Integrator()
            def energy(m):return -1.+.5*.01*(m.q[1,2]-m.q[0,2]-.8)**2
            class Optimizer:
                def run(self,ansatz):
                    self.pre_param=np.zeros(330);ansatz._energy=energy(md.mol);return ansatz
                def get_force(self,*args):
                    d=.01*(md.mol.q[1,2]-md.mol.q[0,2]-.8)
                    return [0.,0.,-d,0.,0.,d]
            def fake_force(name):
                opt=Optimizer();a=opt.run(SimpleNamespace());g=opt.get_force(a);np.savetxt(case/'force_true.txt',g)
            integrators=SimpleNamespace(__file__=str(case/'integrators.py'),VelocityVerlet=Integrator)
            optmod=SimpleNamespace(__file__=str(case/'vqe_opt.py'),optimizer=Optimizer)
            forcemod=SimpleNamespace(__file__=str(case/'force.py'))
            def driver(path,run_name):
                for i in range(p['frames']):md._next();md.time+=md.dt
            oldcwd=Path.cwd();oldpath=list(sys.path);oldargv=list(sys.argv);oldsignals={s:signal.getsignal(s) for s in [signal.SIGTERM,signal.SIGINT]}
            try:
                os.chdir(case);sys.argv=['md_observer.py','--driver',str(case/'md.py'),'--initial',str(case/'initial.json'),
                       '--protocol',str(case/'protocol.json'),'--runtime-lock',str(case/'lock.json')]
                with patch.dict(sys.modules,{'integrators':integrators,'vqe_opt':optmod,'force':forcemod,
                    'nn_input':SimpleNamespace(input_f=lambda r:None),'nn_output':SimpleNamespace(output_f=lambda s:None),
                    'atom_force':SimpleNamespace(force_f=fake_force)}),patch.object(b,'__file__',str(case/'state_bridge.py')), \
                    patch.object(md_observer.runpy,'run_path',side_effect=driver):
                    md_observer.main()
                rows,issue=c.read_observations(case/'observations.jsonl',p,True)
                self.assertEqual(len(rows),21);self.assertIsNone(issue);self.assertTrue(c.load(case/'initial_check.json')['passed'])
                self.assertTrue(all(c.load(case/'provenance.json')['runtime_hashes_unchanged'].values()))
                expected=simulate()
                for a,z in zip(rows,expected):
                    self.assertLess(c.maxdiff(a['coordinates_bohr'],z['coordinates_bohr']),1e-14)
                    self.assertLess(c.maxdiff(a['velocities_au'],z['velocities_au']),1e-14)
                    self.assertEqual(a['MD_Epot_Eh'],a['E_pred_Eh'])
            finally:
                os.chdir(oldcwd);sys.path[:]=oldpath;sys.argv[:]=oldargv
                for signum,handler in oldsignals.items():signal.signal(signum,handler)


class Launcher(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.package=self.root/'package';self.package.mkdir();self.bin=self.root/'bin';self.bin.mkdir()
        for n in ['submit_slurm_queue.sh','run_nve.slurm']:shutil.copyfile(PKG/n,self.package/n)
        py=self.bin/'fakepython';py.write_text('#!/bin/bash\nexit 0\n');py.chmod(0o755)
        self.env=os.environ.copy();self.env.update(PATH=str(self.bin)+':'+self.env.get('PATH',''),QCML_MD_PYTHON=str(py))
    def tearDown(self):self.tmp.cleanup()
    def scheduler(self,text,code=0):
        f=self.bin/'sbatch';f.write_text('#!/bin/bash\nprintf "called\\n" >> "'+str(self.root/'calls')+'"\nprintf "%s\\n" "'+text+'"\nexit '+str(code)+'\n');f.chmod(0o755)
    def launch(self):return subprocess.run(['bash',str(self.package/'submit_slurm_queue.sh')],cwd=str(self.package),env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=10)
    def test_accepted_once(self):
        self.scheduler('12345;testcluster');r=self.launch();self.assertEqual(r.returncode,0)
        self.assertEqual((self.package/'submitted_jobid.txt').read_text().strip(),'12345;testcluster')
        self.assertEqual((self.root/'calls').read_text().count('called'),1)
    def test_real_scheduler_rejection_kept(self):
        self.scheduler('QOSMaxSubmitJobPerUserLimit',1);r=self.launch();self.assertEqual(r.returncode,1)
        self.assertFalse((self.package/'submitted_jobid.txt').exists());self.assertEqual((self.root/'calls').read_text().count('called'),1)
    def test_duplicate_not_resubmitted(self):
        self.scheduler('12345');self.assertEqual(self.launch().returncode,0);self.assertEqual(self.launch().returncode,2)
        self.assertEqual((self.root/'calls').read_text().count('called'),1)
    def test_uncertain_success_blocks_retry(self):
        self.scheduler('unfamiliar successful response');self.assertEqual(self.launch().returncode,2)
        self.assertTrue((self.package/'submitted_jobid.txt').exists());self.assertEqual(self.launch().returncode,2)
        self.assertEqual((self.root/'calls').read_text().count('called'),1)

if __name__=='__main__':unittest.main()
