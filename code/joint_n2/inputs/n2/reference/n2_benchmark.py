#!/usr/bin/env python3
"""N2 frozen-core CASCI and exact-exponential spin-adapted UCC pilot.

The qsector operator algebra, normalization, and exponential-of-sum gradient
are reused unchanged from the preceding QCML reference calculations.
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

HERE = Path(__file__).resolve().parent


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locked_config(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    cfg = load(path)
    if cfg.get('schema') != 'qcml_n2_pilot_v1':
        raise ValueError('Unknown configuration schema')
    if (cfg['ncore'], cfg['ncas'], cfg['nelecas']) != (2, 8, [5, 5]):
        raise ValueError('This validated N2 pilot requires frozen-core CAS(10,8)')
    if cfg['ansatz'] not in ('uccsd', 'uccgsd', 'kupccgsd'):
        raise ValueError('Unknown ansatz family')
    if len(cfg['bond_lengths_angstrom']) != len(set(cfg['bond_lengths_angstrom'])):
        raise ValueError('Duplicate bond lengths')
    root.mkdir(parents=True, exist_ok=True)
    cfgfile = root / 'config.json'
    if cfgfile.exists() and load(cfgfile) != cfg:
        raise ValueError('Output root belongs to another configuration; use another root')
    if not cfgfile.exists():
        dump(cfgfile, cfg)
    return cfg, root


def setup_threads():
    from pyscf import lib
    try:
        lib.current_memory()
    except FileNotFoundError:
        # Some container PID namespaces do not expose /proc/self/statm.
        # Report the process peak RSS as a conservative current-memory estimate.
        import resource
        lib.current_memory = lambda: (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024., 0.)
    threads = int(os.environ.get('QCML_THREADS', '1'))
    lib.num_threads(threads)
    return threads


def runtime():
    import pyscf, scipy
    return {'python': sys.version, 'pyscf': pyscf.__version__, 'numpy': np.__version__,
            'scipy': scipy.__version__, 'platform': platform.platform(),
            'host': platform.node(), 'QCML_THREADS': setup_threads()}


def orbital_order(mol, mf, previous):
    """Match canonical orbitals within core/occupied/virtual and D2h-irrep blocks.

    Only permutations and signs are used. No CASCI data guide the gauge.
    The occupied subspace is preserved, so the reference remains the same RHF state.
    """
    from pyscf import symm, gto
    C = np.asarray(mf.mo_coeff)
    eps, occ = np.asarray(mf.mo_energy), np.asarray(mf.mo_occ)
    sym = np.asarray(symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, C), dtype=str)
    core = np.argsort(eps)[:2]
    role = np.where(occ > 1, 'occupied', 'virtual').astype('<U8')
    role[core] = 'core'
    if not np.all(occ[core] == 2):
        raise RuntimeError('Frozen 1s orbitals must be doubly occupied')
    if previous is None:
        order = np.array(sorted(range(C.shape[1]), key=lambda i: (
            {'core': 0, 'occupied': 1, 'virtual': 2}[role[i]], str(sym[i]), float(eps[i]))))
        signs = np.ones(len(order))
        for j, k in enumerate(order):
            ix = np.argmax(abs(C[:, k]))
            signs[j] = 1 if C[ix, k] >= 0 else -1
        matched = np.ones(len(order))
    else:
        cross = gto.intor_cross('int1e_ovlp', previous['mol'], mol)
        overlap = previous['C'].T @ cross @ C
        oldkeys = list(zip(previous['role'], previous['sym']))
        keys = list(zip(role, sym))
        order = np.zeros(C.shape[1], dtype=int)
        for key in sorted(set(oldkeys)):
            old = [i for i, k in enumerate(oldkeys) if k == key]
            new = [i for i, k in enumerate(keys) if k == key]
            if len(old) != len(new):
                raise RuntimeError('Orbital role/irrep counts changed: ' + str(key))
            rr, cc = linear_sum_assignment(-abs(overlap[np.ix_(old, new)]))
            for a, b in zip(rr, cc):
                order[old[a]] = new[b]
        diag = overlap[np.arange(len(order)), order]
        signs = np.where(diag >= 0, 1., -1.)
        matched = abs(diag)
    C = C[:, order] * signs[None, :]
    if np.max(abs(C.T @ mf.get_ovlp() @ C - np.eye(len(order)))) > 1e-9:
        raise RuntimeError('MO orthogonality check failed')
    expected = ['core'] * 2 + ['occupied'] * 5 + ['virtual'] * 3
    if role[order].tolist() != expected:
        raise RuntimeError('RHF occupied/active ordering mismatch')
    return {'mol': mol, 'C': C, 'eps': eps[order], 'occ': occ[order],
            'sym': sym[order], 'role': role[order], 'order': order,
            'signs': signs, 'matched_abs_overlap': matched}


def make_references(cfg, root):
    from pyscf import gto, scf, mcscf, fci, ao2mo, symm
    from qsector.sector import Sector, PySCFHamiltonian
    dump(root / 'runtime_reference.json', runtime())
    previous = None
    for index, R in enumerate(cfg['bond_lengths_angstrom']):
        out = root / 'references' / ('r%02d' % index)
        out.mkdir(parents=True, exist_ok=True)
        result = out / 'result.json'
        mol = gto.M(atom=[('N', (0., 0., -R/2)), ('N', (0., 0., R/2))],
                    unit='Angstrom', basis=cfg['basis'], symmetry=cfg['symmetry'],
                    spin=0, charge=0, verbose=0, max_memory=4000)
        if result.exists():
            row = load(result)
            z = np.load(out / 'reference.npz', allow_pickle=False)
            if row['reference_sha256'] != digest(out / 'reference.npz'):
                raise RuntimeError('Reference checksum mismatch')
            previous = {'mol': mol, 'C': z['mo_coeff'], 'role': z['orbital_roles'],
                        'sym': z['orbital_symmetries']}
            print('REFERENCE_REUSED', index, R, flush=True)
            continue
        start = time.perf_counter()
        with (out / 'pyscf.log').open('w', encoding='utf-8') as log:
            mol.stdout = log
            mol.verbose = 4
            mf = scf.RHF(mol)
            mf.conv_tol = 1e-12
            mf.max_cycle = 200
            mf.chkfile = str(out / 'rhf.chk')
            mf.kernel()
            if not mf.converged:
                mf = mf.newton()
                mf.conv_tol = 1e-12
                mf.max_cycle = 200
                mf.kernel()
            if not mf.converged:
                raise RuntimeError('RHF did not converge at R=' + str(R))
            gauge = orbital_order(mol, mf, previous)
            C = gauge['C']
            mc = mcscf.CASCI(mf, cfg['ncas'], tuple(cfg['nelecas']))
            mc.ncore = cfg['ncore']
            h1, ecore = mc.get_h1eff(C)
            eri = ao2mo.restore(1, mc.get_h2eff(C), cfg['ncas'])
            n = cfg['ncas']; ne = tuple(cfg['nelecas'])
            solver = fci.direct_spin1_symm.FCI(mol)
            solver.orbsym = np.array([symm.irrep_name2id(mol.groupname, x) for x in gauge['sym'][2:]])
            solver.wfnsym = symm.irrep_name2id(mol.groupname, 'Ag')
            solver = fci.addons.fix_spin_(solver, shift=.5, ss=0.)
            solver.conv_tol = 1e-12
            solver.conv_tol_residual = cfg['thresholds']['reference_residual_Eh']
            solver._keys = set(solver._keys) | {'conv_tol_residual'}
            solver.max_cycle = 300
            solver.max_space = 80
            energy, ci = solver.kernel(h1, eri, n, ne, ecore=ecore)
            ci = np.asarray(ci)
            sec = Sector(n, *ne)
            ham = PySCFHamiltonian(sec, h1, eri, ecore)
            v = ci.ravel()
            physical_energy = float(v @ ham.apply(v) / (v @ v))
            residual = float(np.linalg.norm(ham.apply(v) - physical_energy * v))
            s2 = ham.spin_square(v)
            hf_energy = float(sec.hf() @ ham.apply(sec.hf()))
            dm = fci.direct_spin1.make_rdm1(ci, n, ne)
            noons = np.linalg.eigvalsh(dm)[::-1]
            checks = {
                'casci_converged': bool(solver.converged),
                'physical_residual': residual < cfg['thresholds']['reference_residual_Eh'],
                'spin_singlet': abs(s2) < cfg['thresholds']['abs_S2'],
                'spin_penalty_zero': abs(float(energy) - physical_energy) < 1e-8,
                'rhf_reproduced': abs(hf_energy - mf.e_tot) < 1e-8,
                'rdm1_trace': abs(float(np.trace(dm)) - sum(ne)) < 1e-8,
                'normalization': abs(float(v @ v) - 1.) < 1e-9,
            }
            checks = {k: bool(v) for k, v in checks.items()}
            if not all(checks.values()):
                dump(out / 'failed_checks.json', {'checks': checks, 'residual_Eh': residual,
                     'S2': s2, 'penalized_energy': float(energy), 'physical_energy': physical_energy,
                     'rhf_energy': float(mf.e_tot), 'reconstructed_rhf_energy': hf_energy})
                raise RuntimeError('Reference quality checks failed: ' + str(load(out / 'failed_checks.json')))
            np.savez_compressed(out / 'reference.npz', h1_active=h1, eri_active=eri,
                                core_constant_Eh=ecore, ci=ci, mo_coeff=C,
                                orbital_energies_Eh=gauge['eps'], orbital_occupations=gauge['occ'],
                                orbital_symmetries=gauge['sym'], orbital_roles=gauge['role'],
                                orbital_permutation=gauge['order'], orbital_signs=gauge['signs'],
                                matched_abs_overlap=gauge['matched_abs_overlap'], natural_occupations=noons,
                                active_rdm1=dm, R_angstrom=R)
            row = {'index': index, 'R_angstrom': R, 'basis': cfg['basis'],
                   'ncore': cfg['ncore'], 'n_active_electrons': sum(ne), 'n_active_spatial_orbitals': n,
                   'qubits_before_tapering': 2*n, 'fixed_Ms_dimension': sec.dimension,
                   'E_RHF_Eh': float(mf.e_tot), 'E_CASCI_Eh': physical_energy,
                   'CASCI_S2': s2, 'residual_Eh': residual,
                   'CASCI_internal_residual_tolerance_Eh': solver.conv_tol_residual,
                   'natural_occupations': noons.tolist(),
                   'occupation_nonidempotency': float(np.sum(noons*(2-noons))),
                   'RHF_weight_in_CASCI': float(v[0]**2/(v@v)),
                   'HOMO_Eh': float(np.max(gauge['eps'][gauge['occ'] > 1])),
                   'LUMO_Eh': float(np.min(gauge['eps'][gauge['occ'] < 1])),
                   'min_matched_orbital_abs_overlap': float(np.min(gauge['matched_abs_overlap'])),
                   'orbital_labels': gauge['sym'].tolist(), 'checks': checks,
                   'reference_sha256': digest(out / 'reference.npz'),
                   'seconds': time.perf_counter()-start,
                   'reference_scope': 'Frozen-core CASCI; not full-basis all-electron FCI',
                   'RHF_external_stability_tested': False,
                   'CASCI_used_for_orbital_gauge_or_model_features': False}
            dump(result, row)
            previous = gauge
            print('REFERENCE_DONE', index, R, 'E=', physical_energy,
                  'NOON=', np.round(noons, 5).tolist(), flush=True)


def build_ansatz(cfg, root, index):
    from qsector.algebra import pool
    from qsector.sector import Sector, PySCFHamiltonian, CompiledPool
    from qsector.ansatz import Ansatz
    ref = root / 'references' / ('r%02d' % index)
    row = load(ref / 'result.json')
    if row['reference_sha256'] != digest(ref / 'reference.npz'):
        raise RuntimeError('Reference checksum mismatch')
    z = np.load(ref / 'reference.npz', allow_pickle=False)
    sec = Sector(cfg['ncas'], *cfg['nelecas'])
    ops, labels = pool(cfg['ncas'], cfg['nelecas'][0], cfg['ansatz'])
    cp = CompiledPool(sec, ops)
    ham = PySCFHamiltonian(sec, z['h1_active'], z['eri_active'], float(z['core_constant_Eh']))
    model = Ansatz(cp, ham, cfg['layers'])
    return model, ham, row, z['ci'].ravel(), labels


def gradient_check(cfg, root, index):
    setup_threads()
    a, ham, row, ci, labels = build_ansatz(cfg, root, index)
    rng = np.random.default_rng(826)
    x = rng.normal(0, .02, a.nparams)
    start = time.perf_counter()
    e, grad = a.value_grad(x)
    seconds = time.perf_counter()-start
    d = rng.normal(size=a.nparams); d /= np.linalg.norm(d)
    eps = 1e-5
    fd = (a.energy(x + eps*d) - a.energy(x - eps*d))/(2*eps)
    analytic = float(grad @ d)
    out = {'n_parameters': a.nparams, 'dimension': a.pool.sector.dimension,
           'pool_entries': a.pool.entries, 'energy_Eh': e, 'value_gradient_seconds': seconds,
           'directional_gradient': analytic, 'finite_difference_gradient': fd,
           'gradient_abs_error': abs(analytic-fd), 'passed': abs(analytic-fd)<1e-6,
           'norm2': float(a.state(x) @ a.state(x)), 'S2': ham.spin_square(a.state(x)),
           'quadrature': a.last_quadrature}
    dump(root / ('gradient_check_r%02d.json' % index), out)
    print(json.dumps(out, indent=2))
    if not out['passed']:
        raise RuntimeError('Directional gradient verification failed')


def run_vqe(cfg, root, index, seed):
    setup_threads()
    out = root / 'vqe' / ('r%02d_seed%d' % (index, seed))
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        print('VQE_REUSED', index, seed, flush=True)
        return
    lock = out / 'RUNNING.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('A running/stale lock exists; inspect it before retrying: ' + str(lock))
    with os.fdopen(fd, 'w') as f:
        json.dump({'pid': os.getpid(), 'host': platform.node()}, f)
    start = time.perf_counter()
    try:
        a, ham, ref, ci, labels = build_ansatz(cfg, root, index)
        dump(out / 'runtime.json', runtime())
        theta0 = np.random.default_rng(seed).normal(0, cfg['init_scale'], a.nparams)
        np.save(out / 'initial_theta.npy', theta0, allow_pickle=False)
        count = 0
        with (out / 'history.tsv').open('w', encoding='utf-8', newline='') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(['evaluation', 'seconds', 'E_Eh', 'gradient_inf', 'quadrature_order'])
            def objective(x):
                nonlocal count
                e, g = a.value_grad(x)
                count += 1
                if not np.isfinite(e) or not np.isfinite(g).all():
                    raise RuntimeError('Nonfinite VQE objective')
                w.writerow([count, time.perf_counter()-start, e, float(abs(g).max()),
                            max(a.last_quadrature['orders_last_to_first'])])
                f.flush()
                return e, g
            def callback(x):
                np.save(out / 'latest_theta.npy', x, allow_pickle=False)
            opt = minimize(objective, theta0, method='L-BFGS-B', jac=True,
                           callback=callback, options=cfg['optimizer'])
        e, grad = a.value_grad(opt.x)
        state = a.state(opt.x)
        norm = float(state @ state)
        fidelity = float(abs(state @ ci)**2/(norm * (ci @ ci)))
        s2 = ham.spin_square(state)
        dm = ham.fci.direct_spin1.make_rdm1(state.reshape(ham.s.shape), ham.s.norb, ham.nelec)
        noons = np.linalg.eigvalsh(dm)[::-1]
        delta = e - ref['E_CASCI_Eh']
        numeric = {'finite': bool(np.isfinite(e) and np.isfinite(state).all()),
                   'norm': abs(norm-1)<1e-8, 'spin_singlet': abs(s2)<cfg['thresholds']['abs_S2'],
                   'variational': delta >= -1e-7}
        numerical_passed = all(numeric.values())
        converged = bool(opt.success and abs(grad).max()<cfg['thresholds']['vqe_gradient_inf'])
        np.savez_compressed(out / 'optimized.npz', theta=opt.x, state_ci=state,
                            natural_occupations=noons)
        row = {'index': index, 'R_angstrom': ref['R_angstrom'], 'seed': seed,
               'family': cfg['ansatz'], 'layers': cfg['layers'], 'n_parameters': a.nparams,
               'E_VQE_Eh': float(e), 'E_CASCI_Eh': ref['E_CASCI_Eh'],
               'delta_Eh': float(delta), 'delta_mHa': float(delta*1000),
               'fidelity_to_CASCI': fidelity, 'norm2': norm, 'S2': s2,
               'gradient_inf': float(abs(grad).max()), 'iterations': int(opt.nit),
               'evaluations': int(opt.nfev), 'optimizer_success': bool(opt.success),
               'optimizer_message': str(opt.message), 'converged': converged,
               'numerical_checks': numeric, 'numerical_checks_passed': numerical_passed,
               'within_1p6_mHa': bool(numerical_passed and delta <= cfg['thresholds']['chemical_accuracy_Eh']),
               'natural_occupations': noons.tolist(), 'seconds': time.perf_counter()-start,
               'cross_geometry_warm_start': False, 'CASCI_used_for_initialization': False,
               'reference_sha256': ref['reference_sha256'],
               'optimized_sha256': digest(out / 'optimized.npz'),
               'gradient_quadrature': a.last_quadrature}
        dump(out / 'result.json', row)
        print('VQE_DONE', index, seed, 'gap_mHa=', row['delta_mHa'],
              'F=', fidelity, 'converged=', converged, 'seconds=', row['seconds'], flush=True)
    except BaseException:
        (out / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        lock.unlink(missing_ok=True)


def collect(cfg, root):
    refs = [load(p) for p in sorted((root/'references').glob('r*/result.json'))]
    rows = [load(p) for p in sorted((root/'vqe').glob('r*_seed*/result.json'))]
    polished = [load(p) for p in sorted((root/'polish').glob('r*_seed*/result.json'))]
    out = root / 'summary'; out.mkdir(exist_ok=True)
    def csvout(name, data, keys):
        with (out/name).open('w', encoding='utf-8', newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(data)
    csvout('reference_summary.csv', refs, ['index','R_angstrom','E_RHF_Eh','E_CASCI_Eh',
           'RHF_weight_in_CASCI','occupation_nonidempotency','CASCI_S2','residual_Eh','seconds'])
    csvout('vqe_summary.csv', rows, ['index','R_angstrom','seed','family','layers','n_parameters',
           'E_VQE_Eh','E_CASCI_Eh','delta_mHa','fidelity_to_CASCI','S2','gradient_inf',
           'converged','within_1p6_mHa','seconds'])
    if polished:
        csvout('polished_summary.csv',polished,['index','R_angstrom','seed','family','n_parameters',
               'delta_mHa','fidelity_to_CASCI','gradient_inf','converged','seconds',
               'total_reference_optimization_seconds','total_iterations_including_initial_stage'])
    pairs=[]
    for index in range(len(cfg['bond_lengths_angstrom'])):
        group=[r for r in rows if r['index']==index]
        for a,b in itertools.combinations(group,2):
            za=np.load(root/'vqe'/('r%02d_seed%d'%(index,a['seed']))/'optimized.npz',allow_pickle=False)
            zb=np.load(root/'vqe'/('r%02d_seed%d'%(index,b['seed']))/'optimized.npz',allow_pickle=False)
            p,q=za['state_ci'],zb['state_ci']; dt=za['theta']-zb['theta']
            pairs.append({'index':index,'R_angstrom':a['R_angstrom'],'seed_a':a['seed'],'seed_b':b['seed'],
                          'energy_difference_mHa':abs(a['E_VQE_Eh']-b['E_VQE_Eh'])*1000,
                          'parameter_RMSE':float(np.sqrt(np.mean(dt**2))),
                          'state_overlap_squared':float(abs(p@q)**2/((p@p)*(q@q)))})
    csvout('initialization_pairs.csv',pairs,['index','R_angstrom','seed_a','seed_b',
           'energy_difference_mHa','parameter_RMSE','state_overlap_squared'])
    expected=len(cfg['bond_lengths_angstrom'])*len(cfg['seeds'])
    status={'references_complete':len(refs),'references_expected':len(cfg['bond_lengths_angstrom']),
            'vqe_complete':len(rows),'vqe_expected':expected,
            'all_vqe_completed':len(rows)==expected,
            'converged_vqe':sum(r['converged'] for r in rows),
            'vqe_within_1p6_mHa':sum(r['within_1p6_mHa'] for r in rows),
            'Transformer_training_performed':False,'prediction_results_available':False,
            'training_labels_automatically_selected':False}
    status['polished_endpoints_complete']=len(polished)
    status['polished_endpoints_converged']=sum(r['converged'] for r in polished)
    if cfg['ansatz']=='uccgsd':
        status['representative_scope']={'planned_tasks':[[1,0],[6,0]],
            'initial_endpoints_complete':all(any(r['index']==i and r['seed']==s for r in rows) for i,s in [(1,0),(6,0)]),
            'polished_endpoints_complete':all(any(r['index']==i and r['seed']==s for r in polished) for i,s in [(1,0),(6,0)])}
    from qsector.algebra import pool
    ops,labels=pool(cfg['ncas'],cfg['nelecas'][0],cfg['ansatz'])
    dump(out/'parameter_map.json',{'family':cfg['ansatz'],'layers':cfg['layers'],
         'coefficient_normalization':'L2 norm of normal-ordered fermion coefficients equals one',
         'base_generators':[{'index':i,'label':label,'terms':[{'word':[list(q) for q in w],
             'coefficient':float(c)} for w,c in sorted(op.terms.items())]} for i,(op,label) in enumerate(zip(ops,labels))]})
    dump(out/'status.json',status)
    dump(out/'reference_rows.json',refs)
    if refs:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,
                             'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
        fig,axes=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
        R=np.array([r['R_angstrom'] for r in refs]); es=np.array([r['E_CASCI_Eh'] for r in refs])
        axes[0].plot(R,(np.array([r['E_RHF_Eh'] for r in refs])-es)*1000,'o-',label='RHF error')
        if rows:
            for seed in cfg['seeds']:
                group=sorted([r for r in rows if r['seed']==seed],key=lambda x:x['R_angstrom'])
                if group:
                    axes[0].plot([r['R_angstrom'] for r in group],[r['delta_mHa'] for r in group],
                                 'o--',label='VQE seed %d'%seed,ms=4)
        axes[0].set_ylabel('Energy error relative to CASCI (mHa)')
        axes[0].legend(frameon=False,fontsize=8)
        no=np.array([r['natural_occupations'] for r in refs])
        for i in range(no.shape[1]):
            axes[1].plot(R,no[:,i],'-',lw=1.4)
        axes[1].set_ylabel('CASCI natural orbital occupation')
        axes[1].set_ylim(-.05,2.05)
        axes[2].plot(R,[r['RHF_weight_in_CASCI'] for r in refs],'o-',color='#2463a6')
        axes[2].set_ylabel('RHF determinant weight in CASCI')
        axes[2].set_ylim(-.03,1.03)
        for j,ax in enumerate(axes):
            ax.set_xlabel('N–N distance (Å)')
            ax.text(-.14,1.03,'('+chr(97+j)+')',transform=ax.transAxes,fontweight='bold')
        fig.savefig(out/'N2_reference_and_VQE_diagnostics.png',dpi=240)
        fig.savefig(out/'N2_reference_and_VQE_diagnostics.pdf')
        plt.close(fig)
    print(json.dumps(status,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['references','gradient-check','vqe','collect'])
    ap.add_argument('--config',default=str(HERE/'configs/pilot.json'))
    ap.add_argument('--root',default=str(HERE/'runs/pilot'))
    ap.add_argument('--index',type=int,default=None)
    ap.add_argument('--seed',type=int,default=None)
    args=ap.parse_args();cfg,root=locked_config(args.config,args.root)
    if args.command=='references':make_references(cfg,root)
    elif args.command=='gradient-check':gradient_check(cfg,root,1 if args.index is None else args.index)
    elif args.command=='vqe':
        indices=range(len(cfg['bond_lengths_angstrom'])) if args.index is None else [args.index]
        seeds=cfg['seeds'] if args.seed is None else [args.seed]
        for i in indices:
            if not 0<=i<len(cfg['bond_lengths_angstrom']):raise ValueError('Invalid index')
            for seed in seeds:
                if seed not in cfg['seeds']:raise ValueError('Seed is not in frozen configuration')
                run_vqe(cfg,root,i,seed)
    else:collect(cfg,root)


if __name__=='__main__':
    main()
