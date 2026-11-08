from openfermion import FermionOperator, normal_ordered
from openfermion import jordan_wigner, bravyi_kitaev, symmetry_conserving_bravyi_kitaev
from openfermion.linalg import get_sparse_operator
import scipy
from pyscf import gto, scf, fci, mcscf, dmrgscf
import numpy as np
import os
from functools import reduce

def s2_operator(norb):
    '''
    '''
    Sx = FermionOperator()
    for i in range(norb):
        Sx += FermionOperator(((2*i,1),(2*i+1,0)),0.5) + FermionOperator(((2*i+1,1),(2*i,0)),0.5)

    Sy = FermionOperator()
    for i in range(norb):
        Sy += FermionOperator(((2*i,1),(2*i+1,0)),-0.5*1j) + FermionOperator(((2*i+1,1),(2*i,0)),0.5*1j)

    Sz = FermionOperator()
    for i in range(norb):
        Sz += FermionOperator(((2*i,1),(2*i,0)),0.5) - FermionOperator(((2*i+1,1),(2*i+1,0)),0.5)

    S2 = Sx*Sx + Sy*Sy + Sz*Sz

    return S2, Sz
    #return normal_ordered(S2), normal_ordered(Sz)


def s2_matrix_operator(norb):
    '''
    '''
    S2, Sz = s2_operator(norb)

    return get_sparse_operator(S2)


def number_operator(norb):
    '''
    '''
    N = FermionOperator()
    for i in range(norb):
        N += FermionOperator(((2*i,1),(2*i,0)),1.0) + FermionOperator(((2*i+1,1),(2*i+1,0)),1.0)

    return N


def mapping(opA, imap, nqubit, nelec):
    '''
    '''
    if imap == 'JW':
        qubit_opA = jordan_wigner(opA)
    elif imap == 'BK':
        qubit_opA = bravyi_kitaev(opA, int(nqubit))
    elif imap == 'SCBK':
        qubit_opA = symmetry_conserving_bravyi_kitaev(opA, int(nqubit), int(nelec))

    return qubit_opA


def csc_zeros(nqubit):
    ndim = 2**nqubit
    return scipy.sparse.csc_matrix((ndim,ndim), dtype=complex)


def csc_identity(nqubit):
    ndim = 2**nqubit
    return scipy.sparse.identity(ndim, format='csc', dtype=complex)
    

def write_ansatz(options, ansatz):
    with open('ansatz.txt','wt') as out_file:
        out_file.write(options['ops']['class']+'\n')
        out_file.write(options['ansatz']['form']+'\n')

        group = ansatz._params_group
        for i in range(len(group)):
            if group[i]:
                out_file.write(str(group[i][0])+' '+str(group[i][1])
                                         +' '+str(group[i][2]))
            if i < len(group)-1:
                out_file.write(';')
        out_file.write('\n')

        ops = ansatz._ops
        for m in range(len(ops)):
            for i in ops[m]:
                out_file.write(str(i)+' ')
            if m < len(ops)-1:
                out_file.write(';')

def read_ansatz():
    try:
        with open('ansatz.txt','rt') as in_file:
            ansatz = {}
            ansatz['class'] = str(in_file.readline().strip('\n'))
            ansatz['form'] = str(in_file.readline().strip('\n'))

            ansatz['group'] = []
            group = in_file.readline().strip('\n').split(';')
            for element in group:
                a = element.split()
                if str(a[1]).lower() == 'false':
                    a[1] = False
                else:
                    a[1] = True
                ansatz['group'].append([int(a[0]),a[1],str(a[2])])

            ansatz['ops_list'] = []
            ops_list = in_file.readline().strip('\n').split(';')
            for element in ops_list:
                ansatz['ops_list'].append([int(i) for i in element.split()])
    except OSError as err:
        print("OS error: {0}".format(err))
        
    return ansatz

def integrals_act(mol, ncore=0, ncas=None, mo_coeff=None):
    ''' Energy of nuclear and core electrons and 
        one- and two-electron integrals in the active space
    '''
    if mo_coeff is None: 
        myhf = scf.RHF(mol).run()
        mo_coeff = myhf.mo_coeff

    nmo = mo_coeff.shape[1]
    n_a, n_b = mol.nelec   # number of alpha and beta orbitals
    nocc = n_a - ncore     # number of active occupied orbitals 

    E_nuc = gto.Mole.energy_nuc(mol)
    T = mol.intor('int1e_kin_sph')
    V = mol.intor('int1e_nuc_sph')
    hcore = T + V
    Cocc = mo_coeff[:,0:ncore]
    dm = Cocc @ Cocc.T
    j, k = scf.hf.get_jk(mol, dm)
    ecore = np.trace(2*dm @ (hcore + j - .5*k))
    E_nuc += ecore         # number of nuclear and core electron Coulomb energy

    if ncas is None: ncas = nmo - ncore
    C_act = mo_coeff[:,ncore:ncore+ncas]
    # one-electron integrals in the active space
    h = reduce(np.dot, (C_act.conj().T, hcore + 2*j - k, C_act))
    # two-electron integrals in the active space
    g = mol.intor('int2e_sph')
    g = np.einsum("pqrs,pl->lqrs",g,C_act)
    g = np.einsum("lqrs,qm->lmrs",g,C_act)
    g = np.einsum("lmrs,rn->lmns",g,C_act)
    g = np.einsum("lmns,so->lmno",g,C_act)

    return E_nuc, h, g


def fci_from_hci(mol, ncore, ncas=None, mo_coeff=None):
    ''' Reference energy from hci
    '''
    E_nuc, h, g = integrals_act(mol, ncore, ncas, mo_coeff)
    if ncas is None: ncas = h.shape[1]
    n_a, n_b = mol.nelec   # number of alpha and beta orbitals
    nocc = n_a - ncore     # number of active occupied orbitals 

    #cisolver = hci.SCI(mol)
    #cisolver.ci_coeff_cutoff=1e-4
    #cisolver.select_cutoff=1e-4
    #cisolver.conv_ndet_tol=1e-4
    e, civec = fci.selected_ci.kernel(h, g, ncas, nocc*2, verbose=0)

    return e+E_nuc


def fci_from_casci(mol, ncore, ncas=None, mo_coeff=None):
    ''' Reference energy from casci
    '''
    E_nuc, h1, h2 = integrals_act(mol, ncore, ncas, mo_coeff)
    if ncas is None: ncas = h.shape[1]
    n_a, n_b = mol.nelec   # number of alpha and beta orbitals
    noa = n_a - ncore     # number of active occupied orbitals 
    nob = n_b - ncore     # number of active occupied orbitals 
   
    e, fcivec = fci.direct_spin1.kernel(h1, h2, ncas, (noa,nob), nroots=2,
                                    max_space=30, max_cycle=100)

    return e[0]+E_nuc

def fci_from_dmrg(mf, ncas, nelec):
    '''
    '''
    mc = dmrgscf.DMRGSCF(mf, ncas, nelec)
    mc.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 20))
    mc.fcisolver.memory = 100
    emc = mc.kernel()[0]
    return emc
