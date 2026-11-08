"""One prediction and its unchanged six-component gradient per MD frame.
No energy fitting, force averaging, parameter optimization, or new electronic evaluations.
"""
import hashlib
import numpy as np
from common import require,finite,maxdiff

_protocol=None
_record=None
_count=0

def configure(protocol):
    global _protocol,_record,_count
    _protocol=protocol;_record=None;_count=0

def real_scalar(value,name):
    a=np.asarray(value)
    require(a.size==1,name+' is not a scalar')
    x=complex(a.reshape(-1)[0])
    require(np.isfinite(x.real) and np.isfinite(x.imag),name+' is nonfinite')
    require(abs(x.imag)<=1e-10,name+' has an unexpected imaginary part')
    return float(x.real)

def parameter_hash(parameters):
    t=np.asarray(parameters,dtype=np.float64)
    require(t.shape==(330,) and np.all(np.isfinite(t)),'Expected original 330 finite parameters')
    return hashlib.sha256(t.tobytes(order='C')).hexdigest()

def record_prediction(energy,parameters):
    global _record
    require(_protocol is not None,'Bridge not configured')
    require(_record is None,'More than one prediction before consumption')
    _record={'phase':'predicted','E_pred_Eh':real_scalar(energy,'predicted energy'),
             'parameter_sha256_float64':parameter_hash(parameters)}

def record_gradient(value,parameters):
    require(_record is not None and _record['phase']=='predicted','Gradient without unique prediction')
    require(parameter_hash(parameters)==_record['parameter_sha256_float64'],'Parameters changed during gradient evaluation')
    a=np.asarray(value,dtype=np.complex128)
    require(a.size==6 and a.ndim<=2 and np.all(np.isfinite(a)),'Expected six finite gradient entries')
    require(float(np.max(np.abs(a.imag)))<=1e-10,'Complex gradient is not silently truncated')
    a=np.array(a.real.reshape(2,3),dtype=float)
    require(np.max(np.abs(a.sum(axis=0)))<=_protocol['net_gradient_guard_Eh_per_bohr'],
            'Unexpected net gradient; keep evidence, do not symmetrize or continue')
    _record['returned_gradient_Eh_per_bohr']=a.tolist();_record['phase']='gradient_ready'

def consume(mol,scanner_energy,file_gradient):
    """Called instead of copying Li-z and negating it for H."""
    require(_record is not None and _record['phase']=='gradient_ready','Missing or duplicate prediction/gradient pair')
    q=np.asarray(mol.atom_coords(),dtype=float)
    require(q.shape==(2,3) and np.all(np.isfinite(q)),'Only two-atom LiH is supported')
    require([mol.atom_symbol(i) for i in range(2)]==['Li','H'],'Wrong atom order')
    d=q[1]-q[0]
    require(max(abs(d[0]),abs(d[1]))<=_protocol['axial_geometry_tolerance_bohr'] and d[2]>0,
            'This legacy evaluator is axial Li-left/H-right only; no silent coordinate projection')
    r=float(np.linalg.norm(d))*_protocol['BOHR_ANGSTROM']
    lo,hi=_protocol['legacy_reference_R_range_A']
    require(lo<=r<=hi,'Trajectory left saved orbital/context range; no extrapolation fallback')
    g=np.asarray(_record['returned_gradient_Eh_per_bohr'],dtype=float)
    fg=np.asarray(file_gradient,dtype=float)
    require(fg.size==6 and np.all(np.isfinite(fg)),'Invalid force_true.txt')
    require(maxdiff(fg.reshape(2,3).tolist(),g.tolist())<1e-12,'Disk gradient and same-step returned gradient differ')
    _record['scanner_Epot_Eh']=real_scalar(scanner_energy,'reference scanner energy')
    _record['R_angstrom']=r;_record['phase']='consumed'
    return float(_record['E_pred_Eh']),g.copy()

def peek():
    require(_record is not None and _record['phase']=='consumed','No complete prediction for current frame')
    return dict(_record)

def finish_frame():
    global _record,_count
    require(_record is not None and _record['phase']=='consumed','Invalid frame completion')
    _record=None;_count+=1

def completed_frames():return _count
