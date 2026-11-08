#!/usr/bin/env python3
"""Run the isolated corrected-gradient MD; log same-frame high-precision arrays."""
import argparse
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import traceback
from common import require,load,dump,sha,maxdiff,validate_observation,kinetic


def check_initial(row,baseline,protocol):
    tests=[]
    for k,tol in [('E_pred_Eh',protocol['initial_energy_tolerance_Eh']),('Ekin_Eh',1e-10),('scanner_Epot_Eh',1e-9),
                   ('coordinates_bohr',2e-8),('velocities_au',2e-10),('masses_au',1e-6)]:
        error=maxdiff(row[k],baseline[k]);tests.append({'field':k,'max_abs_error':error,'atol':tol,'passed':error<=tol})
    error=maxdiff(row['returned_gradient_Eh_per_bohr'],protocol['initial_corrected_gradient_Eh_per_bohr'])
    tests.append({'field':'corrected_gradient_vs_previous_same_state_AB','max_abs_error':error,
                  'atol':protocol['initial_gradient_tolerance_Eh_per_bohr'],'passed':error<=protocol['initial_gradient_tolerance_Eh_per_bohr']})
    error=abs(row['E_pred_Eh']-protocol['initial_prediction_Eh'])
    tests.append({'field':'initial_prediction_vs_saved_AB','max_abs_error':error,'atol':1e-9,'passed':error<=1e-9})
    return {'passed':all(t['passed'] for t in tests),'checks':tests,
            'old_applied_gradient_not_required_equal_to_corrected_gradient':True,
            'purpose':'same original initial q/v/m/model/energy; corrected gradient is expected to differ'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--driver',required=True);p.add_argument('--initial',required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--runtime-lock',required=True)
    a=p.parse_args();driver=Path(a.driver).resolve();case=driver.parent
    require(Path.cwd().resolve()==case,'Must execute in new isolated case')
    require(driver==case/'md.py','Unexpected driver')
    sys.path.insert(0,str(case))
    protocol=load(a.protocol);expected=load(a.initial);runtime=load(a.runtime_lock)
    for name,digest in runtime.items():require(sha(case/name)==digest,'Runtime file changed: '+name)
    import numpy as np
    import integrators,vqe_opt,force,state_bridge
    for name,m in [('integrators',integrators),('vqe_opt',vqe_opt),('force',force),('state_bridge',state_bridge)]:
        require(Path(m.__file__).resolve()==case/(name+'.py'),'Wrong imported module: '+name)
    require(sha(force.__file__)==protocol['force_corrected_sha256'],'Not the audited single-line corrected force')
    state_bridge.configure(protocol)
    original_run=vqe_opt.optimizer.run; original_force=vqe_opt.optimizer.get_force
    original_next=integrators.VelocityVerlet._next
    def captured_run(self,*args,**kwargs):
        ansatz=original_run(self,*args,**kwargs)
        state_bridge.record_prediction(ansatz._energy,self.pre_param)
        return ansatz
    def captured_force(self,*args,**kwargs):
        result=original_force(self,*args,**kwargs)
        state_bridge.record_gradient(result,self.pre_param)
        return result
    fp=(case/'observations.jsonl').open('x',encoding='utf-8',buffering=1)
    count=[0]
    def captured_next(self):
        frame=original_next(self);saved=state_bridge.peek()
        returned=saved['returned_gradient_Eh_per_bohr']
        row={'time_au':float(frame.time),'dt_au':float(self.dt),'E_pred_Eh':saved['E_pred_Eh'],
             'Ekin_Eh':float(frame.ekin),'MD_Epot_Eh':float(frame.epot),'scanner_Epot_Eh':saved['scanner_Epot_Eh'],
             'Etot_pred_Eh':saved['E_pred_Eh']+float(frame.ekin),
             'coordinates_bohr':np.asarray(frame.coord,dtype=float).tolist(),
             'velocities_au':np.asarray(frame.veloc,dtype=float).tolist(),
             'masses_au':np.asarray(self._masses,dtype=float).tolist(),
             'returned_gradient_Eh_per_bohr':returned,
             'applied_gradient_Eh_per_bohr':(-np.asarray(self.accel)*np.asarray(self._masses).reshape(-1,1)).tolist(),
             'parameter_sha256_float64':saved['parameter_sha256_float64']}
        validate_observation(row,count[0],protocol,True)
        if count[0]==0:
            test=check_initial(row,expected,protocol);dump(case/'initial_check.json',test)
            dump(case/'initial_observation.json',row)
            require(test['passed'],'Initial conditions or corrected force do not replay; no further MD step')
        fp.write(json.dumps(row,allow_nan=False)+'\n');count[0]+=1;state_bridge.finish_frame()
        if count[0]==1 or count[0]%200==0 or count[0]==protocol['frames']:
            dump(case/'progress.json',{'frames_completed':count[0],'planned_frames':protocol['frames'],
                'last_time_au':row['time_au'],'last_R_angstrom':saved['R_angstrom'],
                'delta_Etot_Eh':row['Etot_pred_Eh']-expected['E_pred_Eh']-expected['Ekin_Eh'],
                'VQE_optimization_performed':False,'network_training_performed':False})
        return frame
    vqe_opt.optimizer.run=captured_run;vqe_opt.optimizer.get_force=captured_force
    integrators.VelocityVerlet._next=captured_next
    def stop(signum,frame):raise RuntimeError('Interrupted by signal '+str(signum)+'; completed frames retained')
    for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,stop)
    try:
        runpy.run_path(str(driver),run_name='__main__')
        require(count[0]==protocol['frames'],'MD stopped early')
    except BaseException as exc:
        dump(case/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'frames_completed':count[0]})
        raise
    finally:
        fp.close()
        vqe_opt.optimizer.run=original_run;vqe_opt.optimizer.get_force=original_force
        integrators.VelocityVerlet._next=original_next
        checks={k:sha(case/k)==v for k,v in runtime.items()}
        modules={}
        for name,m in list(sys.modules.items()):
            path=getattr(m,'__file__',None)
            if path and Path(path).is_file() and Path(path).resolve().parent==case:
                modules[name]={'path':str(Path(path).resolve()),'sha256':sha(path)}
        dump(case/'provenance.json',{'frames_completed':count[0],'runtime_hashes_unchanged':checks,
            'modules':modules,'VQE_optimization_performed':False,'network_training_performed':False,
            'gradient_symmetrized_or_averaged':False,'MD_epot_is_predicted_energy':True,
            'gradient_formula':'single-line-overlap-correction','old_CASCI_scanner_kept_for_reference_only':True})
        require(all(checks.values()),'Runtime code or immutable model inputs changed')

if __name__=='__main__':main()
