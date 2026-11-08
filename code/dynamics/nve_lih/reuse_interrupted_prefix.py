#!/usr/bin/env python3
"""Read-only 8000-au comparison from valid prefix of an interrupted v1 run.

Never relabel the old 688-fs run as completed. New outputs are isolated.
"""
import argparse
import datetime
import json
from pathlib import Path
import tempfile
from common import require, load, dump, sha, safe_parent, read_observations, scalar_rows, read_table, write_table, statistics, ledger, maxdiff

PACKAGE = Path(__file__).resolve().parent


def reuse(old_suite, out_parent, short_protocol=None):
    old_suite=Path(old_suite).expanduser().resolve()
    require(old_suite.is_dir(),'Interrupted suite not found')
    oldplan=load(old_suite/'plan.json')
    require(oldplan['version']=='corrected_force_nve_v1', 'Expected previous full corrected-force NVE run')
    oldproto=oldplan['protocol']
    
    shortproto=short_protocol or load(PACKAGE/'protocol.json')
    target=int(shortproto['frames']);dt=float(shortproto['dt_au'])
    require(target>1 and target<=int(oldproto['frames']) and dt==float(oldproto['dt_au']), 'Incompatible short window')
    require(int(oldproto['frames'])==int(shortproto['old_baseline_frames']), 'Wrong historical full-run frame count')
    require(abs(float(oldproto['end_time_au'])-float(shortproto['old_baseline_end_time_au']))<1e-9,'Wrong historical full-run duration')
    require(abs((target-1)*dt-float(shortproto['end_time_au']))<1e-9, 'Short protocol duration mismatch')
    out_parent=safe_parent(out_parent,[old_suite,PACKAGE]);out_parent.mkdir(parents=True,exist_ok=True)
    audit=Path(tempfile.mkdtemp(prefix='qcml_aimd_nve_prefix_',dir=str(out_parent)))
    raw=old_suite/'corrected_dt2p5/observations.jsonl'
    require(raw.is_file(),'No recorded frames found in interrupted run')
    obs,issue=read_observations(raw,oldproto,True,True)
    recorded=len(obs)
    prior=load(old_suite/'status.json') if (old_suite/'status.json').is_file() else None
    report={'status':'PREFIX_AVAILABLE' if recorded>=target else 'INSUFFICIENT_PREFIX',
            'original_run_was_interrupted':bool(prior and prior.get('status')=='FAILED_OR_INCOMPLETE'),
            'source_suite':str(old_suite),'source_status':prior['status'] if prior else 'UNKNOWN',
            'source_observation_sha256':sha(raw),'source_valid_frames':recorded,
            'source_terminal_issue':issue,
            'required_frames':target,'fixed_end_time_au':shortproto['end_time_au'],
            'source_full_688fs_completed':False,
            'replayed_MD':False,'force_recomputed':False,'network_called':False,
            'reuse_criteria':'same initial state, same dt, first 3201 valid frames, fixed 0-8000 au window'}
    if recorded:
        report['valid_end_time_au']=float(obs[-1]['time_au'])
        report['valid_end_time_fs']=float(obs[-1]['time_au'])*float(shortproto['AU_TIME_FS'])
    if recorded>=target:
        require((old_suite/'corrected_dt2p5/initial_check.json').is_file(), 'Missing original initial-state check')
        initial=load(old_suite/'corrected_dt2p5/initial_check.json')
        require(bool(initial['passed']), 'Interrupted trajectory initial check had failed')
        reference=load(old_suite/'initial_reference.json')
        require(maxdiff(obs[0]['coordinates_bohr'],reference['coordinates_bohr'])<=2e-8, 'Initial position mismatch')
        require(maxdiff(obs[0]['velocities_au'],reference['velocities_au'])<=2e-10, 'Initial velocity mismatch')
        require(abs(obs[0]['E_pred_Eh']-reference['E_pred_Eh'])<=1e-9,'Initial energy mismatch')
        old=read_table(old_suite/'old_dt2p5.tsv')
        require(len(old)==int(oldproto['frames']),'Old reference table incomplete')
        old=old[:target]
        shortobs=obs[:target]
        new=scalar_rows(shortobs,shortproto)
        require(all(float(a['time_au'])==float(b['time_au']) for a,b in zip(old,new)), 'Time-grid mismatch')
        st_old=statistics(old);st_new=statistics(new)
        if target != len(new):raise AssertionError('Unexpected short prefix length')
        summary={'old_same_window':st_old,'new_same_window':st_new,
                 'ratios_new_over_old':{k:st_new[k]/st_old[k] if st_old[k]!=0 else None for k in ('max_abs_delta_Eh','rms_from_initial_Eh','peak_to_peak_Eh')},
                 'same_initial_conditions':True,'same_duration':True,'time_window_end_au':shortproto['end_time_au'],
                 'uncompleted_long_run_not_reclassified':True,
                 'definition':'E_predicted + same-frame kinetic; no smoothing/detrending/phase adjustment'}
        write_table(audit/'old_prefix.tsv',old)
        write_table(audit/'new_prefix.tsv',new)
        dump(audit/'energy_comparison.json',summary)
        ls,_=ledger(shortobs,shortproto)
        dump(audit/'short_prefix_ledger.json',ls)
        report['old_max_abs_dE_Eh']=st_old['max_abs_delta_Eh']
        report['new_max_abs_dE_Eh']=st_new['max_abs_delta_Eh']
    dump(audit/'status.json',report)
    print('prefix_audit_dir =',audit,flush=True)
    print('prefix_status =',report['status'],flush=True)
    print('validated_frames =',recorded,'required =',target,flush=True)
    print('prefix_report =',audit/'status.json',flush=True)
    return audit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--old-suite',required=True)
    ap.add_argument('--out-parent',required=True)
    a=ap.parse_args()
    reuse(a.old_suite,a.out_parent)

if __name__=='__main__':main()
