#!/usr/bin/env python3
"""Plot actual saved energy time series; never smooth, detrend, or phase-align."""
import argparse
from pathlib import Path
from common import read_table,load


def make(suite):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    suite=Path(suite);old=read_table(suite/'old_dt2p5.tsv');new=read_table(suite/'corrected_dt2p5.tsv')
    protocol=load(suite/'protocol.json');out=suite/'figures';out.mkdir(exist_ok=True)
    old=old[:len(new)]
    records=[(old,'Original gradient / original handoff','--'),(new,'Corrected gradient / full-component handoff','-')]
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.svg'));plt.close(fig)
    for end,name in [(None,'total_energy_full_window'),(60.,'total_energy_first_60fs')]:
        fig,ax=plt.subplots(figsize=(10,5.2))
        for rows,label,style in records:
            rr=[r for r in rows if end is None or float(r['time_fs'])<=end]
            ax.plot([float(r['time_fs']) for r in rr],[float(r['delta_Etot_Eh'])*1000 for r in rr],
                    linestyle=style,linewidth=.9,label=label)
        ax.set_xlabel('Time (fs)');ax.set_ylabel('Change in predicted E + kinetic energy (mHartree)')
        ax.set_title('LiH / original QCML predicted states, dt = 2.5 a.u.\nNo VQE refinement, smoothing, detrending, or phase alignment')
        ax.grid(alpha=.25);ax.legend(fontsize=8.5);save(fig,name)
    fig,ax=plt.subplots(figsize=(10,5.2))
    for rows,label,style in records:
        ax.plot([float(r['time_fs']) for r in rows],[float(r['R_angstrom']) for r in rows],linestyle=style,linewidth=.9,label=label)
    ax.set_xlabel('Time (fs)');ax.set_ylabel('Li-H bond length (Angstrom)');ax.set_title('Same initial coordinates, velocities, masses and time step')
    ax.grid(alpha=.25);ax.legend(fontsize=8.5);save(fig,'bond_length_full_window')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True);a=p.parse_args();print(make(a.suite))
