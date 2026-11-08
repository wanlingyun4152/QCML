"""Read-only inputs and finite scalar diagnostics. Python >=3.8."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

MIB=1024*1024

def require(ok,message):
    if not ok: raise ValueError(message)

def load(path):
    with Path(path).open(encoding='utf-8') as f: return json.load(f)

def json_native(obj):
    # JSON does not encode NumPy bool_/int64 automatically. Preserve numeric types.
    if type(obj).__module__.split('.')[0]=='numpy' and getattr(obj,'shape',None)==() and hasattr(obj,'item'):
        value=obj.item()
        if type(value) in (bool,int,float,str):return value
    raise TypeError('Unsupported JSON value: '+type(obj).__name__)

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False,default=json_native);f.write('\n')
    os.replace(str(tmp),str(path))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(MIB),b''):h.update(b)
    return h.hexdigest()

def flatten(x):
    if isinstance(x,(list,tuple)):
        for y in x:yield from flatten(y)
    else:yield float(x)

def finite(x,name='value'):
    v=list(flatten(x));require(v and all(math.isfinite(t) for t in v),'Nonfinite/empty '+name);return v

def maxdiff(a,b):
    a,b=finite(a),finite(b);require(len(a)==len(b),'Vector size mismatch');return max(abs(x-y) for x,y in zip(a,b))

def kinetic(m,v):
    return .5*math.fsum(float(m[i])*math.fsum(float(x)**2 for x in v[i]) for i in range(len(m)))

def write_table(path,rows):
    rows=list(rows);require(rows,'Cannot write empty table')
    fields=list(rows[0])
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rows)

def read_table(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))

def safe_parent(parent,protected):
    parent=Path(parent).expanduser().resolve()
    for q in protected:
        q=Path(q).expanduser().resolve()
        require(parent!=q and q not in parent.parents,'Output inside protected input: '+str(q))
    return parent

def safe_file(root,rel):
    root=Path(root).resolve();p=Path(rel)
    require(not p.is_absolute() and '..' not in p.parts,'Unsafe relative path')
    dst=root/p
    require(root in dst.resolve().parents,'File escapes input root: '+rel)
    require(not any((root.joinpath(*p.parts[:i])).is_symlink() for i in range(1,len(p.parts)+1)),
            'Symlink within frozen input tree: '+rel)
    require(dst.is_file(),'Missing input: '+str(dst));return dst

def verify_package(package):
    package=Path(package).resolve(); manifest=load(package/'PACKAGE_SHA256.json')
    for rel,digest in manifest.items():
        p=safe_file(package,rel);require(sha(p)==digest,'Package hash mismatch: '+rel)
    return {'checked_files':len(manifest),'passed':True}

def statistics(rows,au_fs=0.02418884254):
    require(len(rows)>=2,'Need at least two completed frames')
    t=[float(r['time_au']) for r in rows]; e=[float(r['Etot_pred_Eh']) for r in rows]
    require(all(t[i+1]>t[i] for i in range(len(t)-1)),'Non-increasing time')
    d=[x-e[0] for x in e];mt=math.fsum(t)/len(t);md=math.fsum(d)/len(d)
    slope=math.fsum((x-mt)*(y-md) for x,y in zip(t,d))/math.fsum((x-mt)**2 for x in t)
    return {'records':len(rows),'time_start_au':t[0],'time_end_au':t[-1],
            'max_abs_delta_Eh':max(map(abs,d)),'rms_from_initial_Eh':math.sqrt(math.fsum(x*x for x in d)/len(d)),
            'end_minus_initial_Eh':d[-1],'peak_to_peak_Eh':max(d)-min(d),
            'linear_slope_Eh_per_au':slope,'linear_slope_mHa_per_ps':slope*1e6/au_fs,
            'definition':'predicted-state potential + same-frame kinetic, initial subtraction only',
            'smoothed':False,'detrended':False,'conservation_pass':None}

def validate_observation(r,index,protocol,corrected=False):
    dt=float(protocol['dt_au'])
    require(abs(float(r['time_au'])-index*dt)<1e-9,'Wrong frame time at '+str(index))
    require(float(r['dt_au'])==dt,'dt changed')
    for k in ['coordinates_bohr','velocities_au','applied_gradient_Eh_per_bohr']:
        require(isinstance(r[k],list) and len(r[k])==2 and all(len(v)==3 for v in r[k]),'Wrong shape: '+k)
        finite(r[k],k)
    require(len(r['masses_au'])==2 and min(finite(r['masses_au']))>0,'Invalid masses')
    for k in ['E_pred_Eh','Ekin_Eh','Etot_pred_Eh','scanner_Epot_Eh']:finite(r[k],k)
    kr=kinetic(r['masses_au'],r['velocities_au'])
    require(abs(kr-float(r['Ekin_Eh']))<1e-10,'Kinetic ledger mismatch')
    require(abs(float(r['Etot_pred_Eh'])-float(r['E_pred_Eh'])-float(r['Ekin_Eh']))<1e-10,'Energy ledger mismatch')
    if corrected:
        require(abs(float(r['MD_Epot_Eh'])-float(r['E_pred_Eh']))<1e-10,'MD Epot is not prediction')
        require(maxdiff(r['returned_gradient_Eh_per_bohr'],r['applied_gradient_Eh_per_bohr'])<protocol['raw_to_applied_gradient_tolerance'],
                'Returned gradient not propagated unchanged')
    return kr

def read_observations(path,protocol,corrected=False,allow_partial=False):
    result=[];issue=None
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            try:
                require(line.endswith('\n'),'Incomplete final JSON line')
                r=json.loads(line);validate_observation(r,len(result),protocol,corrected);result.append(r)
            except Exception as e:
                issue={'first_invalid_record':len(result),'error':repr(e)}
                if not allow_partial:raise
                break
    if not allow_partial:require(len(result)==protocol['frames'],'Incomplete trajectory: '+str(len(result)))
    return result,issue

def scalar_rows(obs,protocol):
    out=[]; initial=float(obs[0]['Etot_pred_Eh']) if obs else 0.
    for r in obs:
        q=r['coordinates_bohr'];g=r['applied_gradient_Eh_per_bohr']
        dist=math.sqrt(math.fsum((q[1][i]-q[0][i])**2 for i in range(3)))
        out.append({'time_au':r['time_au'],'time_fs':r['time_au']*protocol['AU_TIME_FS'],
            'R_angstrom':dist*protocol['BOHR_ANGSTROM'],'E_pred_Eh':r['E_pred_Eh'],
            'Ekin_Eh':r['Ekin_Eh'],'Etot_pred_Eh':r['Etot_pred_Eh'],
            'delta_Etot_Eh':r['Etot_pred_Eh']-initial,'scanner_Epot_Eh':r['scanner_Epot_Eh'],
            'gLi_x':g[0][0],'gLi_y':g[0][1],'gLi_z':g[0][2],
            'gH_x':g[1][0],'gH_y':g[1][1],'gH_z':g[1][2]})
    return out

def ledger(obs,protocol):
    """Check actual recorded Verlet equations; NOT certify force accuracy."""
    require(obs,'No observations')
    m=obs[0]['masses_au'];dt=float(protocol['dt_au']);n=len(obs)
    q0=obs[0]['coordinates_bohr'];v0=obs[0]['velocities_au'];g0=obs[0]['applied_gradient_Eh_per_bohr']
    p0=[math.fsum(m[a]*v0[a][d] for a in range(2)) for d in range(3)]
    c0=[math.fsum(m[a]*q0[a][d] for a in range(2))/sum(m) for d in range(3)]
    mx={k:0. for k in ['K_rebuild_Eh','q_update_bohr','v_update_au','gradient_handoff_Eh_per_bohr',
                       'net_gradient_Eh_per_bohr','momentum_delta_au','COM_linear_residual_bohr','work_kinetic_Eh','work_potential_Eh']}
    rows=[];w=0.;k0=obs[0]['Ekin_Eh'];e0=obs[0]['E_pred_Eh']
    for i,r in enumerate(obs):
        q,v,g=r['coordinates_bohr'],r['velocities_au'],r['applied_gradient_Eh_per_bohr']
        require(r['masses_au']==m,'Mass changed')
        kr=kinetic(m,v);mx['K_rebuild_Eh']=max(mx['K_rebuild_Eh'],abs(kr-r['Ekin_Eh']))
        net=max(abs(g[0][d]+g[1][d]) for d in range(3));mx['net_gradient_Eh_per_bohr']=max(mx['net_gradient_Eh_per_bohr'],net)
        p=[math.fsum(m[a]*v[a][d] for a in range(2)) for d in range(3)]
        c=[math.fsum(m[a]*q[a][d] for a in range(2))/sum(m) for d in range(3)]
        pd=maxdiff(p,p0); cd=maxdiff(c,[c0[d]+p0[d]/sum(m)*r['time_au'] for d in range(3)])
        mx['momentum_delta_au']=max(mx['momentum_delta_au'],pd);mx['COM_linear_residual_bohr']=max(mx['COM_linear_residual_bohr'],cd)
        if 'returned_gradient_Eh_per_bohr' in r:
            mx['gradient_handoff_Eh_per_bohr']=max(mx['gradient_handoff_Eh_per_bohr'],maxdiff(g,r['returned_gradient_Eh_per_bohr']))
        qr=vr=0.
        if i:
            z=obs[i-1];qp,vp,gp=z['coordinates_bohr'],z['velocities_au'],z['applied_gradient_Eh_per_bohr']
            qr=max(abs(q[a][d]-qp[a][d]-dt*vp[a][d]+.5*dt*dt*gp[a][d]/m[a]) for a in range(2) for d in range(3))
            vr=max(abs(v[a][d]-vp[a][d]+.5*dt*(gp[a][d]+g[a][d])/m[a]) for a in range(2) for d in range(3))
            w+=-.5*math.fsum((gp[a][d]+g[a][d])*(q[a][d]-qp[a][d]) for a in range(2) for d in range(3))
        ka=kr-k0-w;eb=r['E_pred_Eh']-e0+w
        for key,val in [('q_update_bohr',qr),('v_update_au',vr),('work_kinetic_Eh',abs(ka)),('work_potential_Eh',abs(eb))]:mx[key]=max(mx[key],val)
        rows.append({'time_au':r['time_au'],'q_update_residual_bohr':qr,'v_update_residual_au':vr,
            'net_gradient_max_Eh_per_bohr':net,'momentum_delta_max_au':pd,'COM_linear_residual_bohr':cd,
            'cumulative_trapezoidal_work_Eh':w,'delta_K_minus_work_Eh':ka,'delta_Epred_plus_work_Eh':eb})
    return {'records':n,'maxima':mx,'work_is_discrete_diagnostic_not_redefined_potential':True},rows
