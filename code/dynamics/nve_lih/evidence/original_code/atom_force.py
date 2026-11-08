import sys
sys.path.append("/home/huangdq_group/liujie/data/taoyx/trans/md_new/src_force_dipole")
from pyscf import gto
from algorithms import *
import math
import os
import numpy as np
import pandas as pd


def test(r):
    mol = gto.Mole()
    mol.atom = [('Li', (0,0,0)), ('H', (0,0,r))]
    mol.charge = 0
    mol.spin = 0
    mol.basis = 'sto-3g'
    mol.symmetry = 1
    mol.build()

    options = {
               'vqe' : {'name':'uccgsd-vqe','algorithm':'fix-vqe'},
               'scf' : {'check_mo_sign' : True, 'mo_list':['1A1', '2A1', '3A1', '1E1x', '1E1y', '4A1']},
               'ops' : {'class':'fermionic','spin_sym':'sa'},
               'ansatz' : {'method':'ucc','form':'unitary'},
               'opt' : {'maxiter':300}
              }

    ansatz = run_vqe(mol, options)

#if __name__== "__main__":
def force_f(file_name):

    pre_data = pd.read_csv(f"{file_name}_clean.txt", sep='\t')
    num_bonds = len(pre_data._stat_axis)
    num_params = int(pre_data.loc[0,'no_parameters'])
#    print(num_bonds, num_params)
    params = np.zeros(shape=[num_bonds, num_params+1])
    params[:,0] = pre_data['bondlengths']
    param_cols = [int(col.split('m')[-1]) for col in pre_data.columns if 'err_VQE_param' in col]
    for j in param_cols:
        params[:,j] = pre_data[f'pre_VQE_param{j}']
            #params[:,j] = pre_data[f'VQE_param{j}']

    for i in range(params.shape[0]):
        r = params[i,0]
#        print(i,r)
#        print("System LiH", "Number =", i, "Bond Length =", r)
        theta = params[i,1:]
        np.savetxt('params.txt', theta)
        mo_r = round(r,2)
        mo_coeff = np.loadtxt(f"/home/huangdq_group/liujie/data/taoyx/trans/vqechem/lih/mo/mo_{mo_r}.txt")
        np.save("mo.npy", mo_coeff)
#        import datetime
#        t3 = datetime.datetime.now()
        test(r)
#        t4 = datetime.datetime.now()
#        print("t3 :", t3)
#        print("t4 :", t4)
#        print("t4 - t3 :", t4 - t3, flush=True)


