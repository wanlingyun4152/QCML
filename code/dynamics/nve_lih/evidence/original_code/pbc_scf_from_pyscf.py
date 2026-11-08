from pyscf.pbc import scf, tools
from openfermion import FermionOperator
from openfermion.linalg import get_sparse_operator
from functools import reduce
from tools import s2_operator, number_operator, mapping
from symmer import PauliwordOp
from symmer.projection import QubitTapering
import numpy as np
import scipy

def get_1e_integral_pbc(self):
    '''CAS sapce one-electron hamiltonian

    Args:
        self : a scfinit object

    Returns:
        A tuple, the first is the effective one-electron hamiltonian defined in CAS space,
        the second is the electronic energy from core.
    '''
    hcore = self._hf.get_hcore(self._mol, self._kpts)
    hcore = np.asarray([reduce(np.dot, (mo.T.conj(), hcore[k], mo))
                               for k, mo in enumerate(self.mo_coeff)])

    nmo = self._nmo
    nkpts = len(self._kpts)
    ncas = self._norb
    h1eff = np.zeros((ncas,ncas),dtype=np.complex128)
    for ki in range(nkpts):
        for i in range(nmo):
            for j in range(nmo):
                p = self._k2m[ki,i]
                q = self._k2m[ki,j]
                h1eff[p,q] = hcore[ki,i,j]/nkpts

    if self._k2g:
        h1eff = np.einsum("qp,qr,rs->ps",self._U.conj(),h1eff,self._U)

    Enuc = self._hf.energy_nuc()

    return h1eff, Enuc

def get_2e_integral_pbc(self):
    ''' CAS space two-electron integral

    Returns:
        two electron integral in MO
    '''
    norb = self._norb
    nmo = self._nmo
    nkpts = len(self._kpts)
    eri_act = np.zeros((norb,norb,norb,norb),dtype=np.complex128)
    for kp in range(nkpts):
        for kq in range(nkpts):
            for kr in range(nkpts):
                ks = self._kconserv[kp,kq,kr]
                eri_kpt = self._hf.with_df.ao2mo([self._hf.mo_coeff[i] for i in (kp,kq,kr,ks)],
                                            [self._kpts[i] for i in (kp,kq,kr,ks)],compact=False)
                eri_kpt = eri_kpt.reshape([nmo]*4)
                for i in range(nmo):
                    for j in range(nmo):
                        for k in range(nmo):
                            for l in range(nmo):
                                p = self._k2m[kp,i]
                                q = self._k2m[kq,j]
                                r = self._k2m[kr,k]
                                s = self._k2m[ks,l]
                                eri_act[p,q,r,s] = eri_kpt[i,j,k,l]/nkpts/nkpts

    if self._k2g:
        eri_act = np.einsum("pqrs,pl->lqrs",eri_act,self._U.conj())
        eri_act = np.einsum("lqrs,qm->lmrs",eri_act,self._U)
        eri_act = np.einsum("lmrs,rn->lmns",eri_act,self._U.conj())
        eri_act = np.einsum("lmns,so->lmno",eri_act,self._U)

    return eri_act


class PySCF_PBC():
    ''' Prepare SCF under periodic boundary condition for VQE using PySCF
    '''
    def __init__(self, mol, options):
        ''' Run PySCF PBC RHF 
        Args:
            mol:   Molecule object
            ncas:  number of active orbitals
            ncore: number of frozen occupied orbitals
        '''
        self._name = 'pbc_scf'
        # mapping from fermion to qubit
        self._mapping = options['scf']['mapping'].upper()
        # orbital symmetry
        self._orbsym = None
        # product table of orbital symmetry
        self._prod_table = None
        # number of active orbitals
        self._ncas = options['scf']['ncas']
        # number of core orbitals
        self._ncore = options['scf']['ncore']
        # list of active orbitals
        mo_list = options['scf']['mo_list']
        # penalty coefficient for spin constraint
        sshift = options['scf']['sshift']
        # penalty coefficient for particle number constraint
        nshift = options['scf']['nshift']
        # Qubit tapering
        self._taperoff = options['scf']['taperoff']
        # check the sign of molecular coefficients
        check_mo_sign = options['scf']['check_mo_sign']
        noa, nob = mol.nelec
        self._nbasis = mol.nao_nr()          # number of basis sets

        self._mol = mol

        nk = options['pbc']['nk']
        scaled_center = options['pbc']['scaled_center']
        k2g = options['pbc']['k2g']        
        kpts = mol.make_kpts(nk,scaled_center=scaled_center)
        self._nk = nk
        self._kpts = kpts
        self._k2g = k2g

        exxdiv = options['pbc']['exxdiv']
        if mol.pseudo is None:
            hf = scf.KRHF(mol,kpts).density_fit()
        else:
            hf = scf.KRHF(mol,kpts,exxdiv=exxdiv)
        hf.kernel()
        self._hf = hf

        self.mo_coeff = hf.mo_coeff
        nao, nmo = hf.mo_coeff[0].shape
        nel = mol.nelectron
        nkpts = len(kpts)
        self._nmo = nmo
        noa, nob = mol.nelec

        self._noa = noa*nkpts
        self._nob = nob*nkpts
        self._norb = self._nmo*nkpts
        self._nva = self._norb - self._noa
        self._nvb = self._norb - self._nob
        self._ncore = 0
        self._ncas = self._norb
        self._nqubit = self._ncas*2

        kconserv = tools.get_kconserv(mol, kpts)
        kcons = np.zeros((nkpts,nkpts,nkpts,nkpts),dtype=int)
        for kp in range(nkpts):
            for kq in range(nkpts):
                for kr in range(nkpts):
                    ks = kconserv[kp,kq,kr]
                    kcons[kp,kq,kr,ks] = 1
                    kcons[kp,ks,kr,kq] = 1
        self._kcons = kcons 
        self._kconserv = kconserv
    
        idx = np.argsort(np.hstack(hf.mo_energy))
        if k2g:
            m2k = [[i//nmo,i%nmo] for i in range(self._norb)]
        else:
            m2k = [[idx[i]//nmo,idx[i]%nmo] for i in range(self._norb)]
        self._m2k = m2k
        k2m = np.zeros((nkpts,self._nmo),dtype=int)
        for i in range(self._norb):
            k2m[m2k[i][0],m2k[i][1]] = i
        self._k2m = k2m
        print([hf.mo_energy[m2k[i][0]][m2k[i][1]] for i in range(self._norb)])

        if self._k2g:
             scell, E_g, C, mo_phase = tools.k2gamma.mo_k2gamma(mol, hf.mo_energy, hf.mo_coeff, kpts)
             U = np.zeros((nkpts*nmo,nkpts*nmo),dtype=np.complex128)
             for k in range(nkpts):
                 for m in range(nmo):
                     U[k2m[k,m],:] = mo_phase[k,m,:]
             self._U = U

        self._h1, self._Enuc = get_1e_integral_pbc(self)
        print("Nuclear energy: %15.8f" %self._Enuc)

        self._fermi_s2, Sz = s2_operator(self._ncas)
        self._qubit_s2 = mapping(self._fermi_s2, self._mapping, self._nqubit, self._noa*2)
        self._s2 = get_sparse_operator(self._qubit_s2)

        self._fermi_N = number_operator(self._ncas)
        self._qubit_N = get_sparse_operator(self._fermi_N)

        self._fermi_ham_bare = self.hamiltonian()
        #self._qubit_ham_bare = mapping(self._fermi_ham_bare, self._mapping, self._nqubit, self._noa*2)
        #self._ham_bare = get_sparse_operator(self._qubit_ham_bare)
        self._ham_bare = get_sparse_operator(self._fermi_ham_bare)

        self._fermi_ham = self._fermi_ham_bare
        if (abs(sshift) > 0):
            self._fermi_ham = self._fermi_ham + sshift/2 * self._fermi_s2 
           
        nelec = self._noa + self._nob
        if (abs(nshift) > 0):
            self._fermi_ham = self._fermi_ham + nshift/2 * (self._fermi_N-nelec)*(self._fermi_N-nelec)

        self._qubit_ham = mapping(self._fermi_ham, self._mapping, self._nqubit, self._noa*2)
        nHq = len(self._qubit_ham.terms)
        self._qubit_ham.compress(1e-15)
        nHqc = len(self._qubit_ham.terms)
        print(f'The number of terms in the qubit Hamiltonin is compressed from {nHq} -> {nHqc}.\n')
        self._ham = get_sparse_operator(self._qubit_ham)
        #self._ham = get_sparse_operator(self._fermi_ham)

        if self._taperoff:
            H = PauliwordOp.from_openfermion(self._qubit_ham)
            QT = QubitTapering(H)
            print(f'Qubit tapering permits a reduction of {H.n_qubits} -> {H.n_qubits-QT.n_taper} qubits.\n')
            print('The following symmetry generators were identified:\n')
            print(QT.symmetry_generators); print()
            print('which we may rotate onto the single-qubit Pauli operators\n')
            print(QT.symmetry_generators.rotate_onto_single_qubit_paulis()); print()
            print('via a sequence of Clifford operations R_k = e^{i pi/4 P_k} where:\n')
            for index, (P_k, angle) in enumerate(QT.symmetry_generators.stabilizer_rotations):
                P_k.sigfig=0
                print(f'P_{index} = {P_k}')

            hf_array = np.zeros(self._nqubit)
            hf_array[::2] = np.hstack([np.ones(self._noa), np.zeros(self._nqubit//2-self._noa)])
            hf_array[1::2] = np.hstack([np.ones(self._nob), np.zeros(self._nqubit//2-self._nob)])
            ref_state=hf_array.astype(int)
            H_taper = QT.taper_it(ref_state=ref_state)
            #print(H_taper)
            print(ref_state)
            self._free_qubit_indices = QT.free_qubit_indices
            ref_taper = ref_state[QT.free_qubit_indices]
            from symmer.symplectic import QuantumState
            print(ref_taper)
            ref_psi_tap = QuantumState([ref_taper])
            print('Expectation value  =', (ref_psi_tap.dagger * H_taper * ref_psi_tap).real)

            from symmer.utils import exact_gs_energy
            gs_nrg, gs_vec = exact_gs_energy(H_taper.to_sparse_matrix)
            print(f'Ground state energy = {gs_nrg}')
            self._qubit_ham = H_taper.to_openfermion
            nHqc = len(self._qubit_ham.terms)
            print(f'The number of terms in the qubit-taper Hamiltonin is reduced from {nHq} -> {nHqc}.\n')

            S2 = PauliwordOp.from_openfermion(self._qubit_s2)
            S2_taper = QT.taper_it(aux_operator=S2)
            self._s2 = S2_taper.to_sparse_matrix
            #self._qubit_s2 = S2_taper.to_openfermion
            #self._s2 = get_sparse_operator(self._qubit_s2)

            Nq = mapping(self._fermi_N, self._mapping, self._nqubit, self._noa*2)
            Nq = PauliwordOp.from_openfermion(Nq)
            Nq_taper = QT.taper_it(aux_operator=Nq)
            self._qubit_N = Nq_taper.to_sparse_matrix
            #self._qubit_N = get_sparse_operator(Nq_taper.to_openfermion)

            Hq = mapping(self._fermi_ham_bare, self._mapping, self._nqubit, self._noa*2)
            Hq = PauliwordOp.from_openfermion(Hq)
            Hq_taper = QT.taper_it(aux_operator=Hq)
            self._ham_bare = Hq_taper.to_sparse_matrix

            self._nqubit = H.n_qubits-QT.n_taper

        self._ham = get_sparse_operator(self._qubit_ham)

        self.print_scf_info()


    def hamiltonian(self):
        ''' Build second quantazied Hamiltonian
        H = \sum_{pq} h1_{pq} a_p^+ a_q 
        + 1/2 \sum_{pqrs} h2_{psqr} a_p^+ a_q^+ a_r a_s
        '''
        nmo = self._norb
        ham = FermionOperator((),self._Enuc)

        for p in range(nmo):
            for q in range(nmo):
                ham += FermionOperator(((2*p,1),(2*q,0)), self._h1[p,q])
                ham += FermionOperator(((2*p+1,1),(2*q+1,0)), self._h1[p,q])

        h2 = get_2e_integral_pbc(self)
        for p in range(nmo):
            for q in range(nmo):
                for r in range(nmo):
                    for s in range(nmo):
                        ham += 0.5*FermionOperator(((2*p,1),(2*q,1),(2*r,0),(2*s,0)), h2[p,s,q,r])
                        ham += 0.5*FermionOperator(((2*p,1),(2*q+1,1),(2*r+1,0),(2*s,0)), h2[p,s,q,r])
                        ham += 0.5*FermionOperator(((2*p+1,1),(2*q,1),(2*r,0),(2*s+1,0)), h2[p,s,q,r])
                        ham += 0.5*FermionOperator(((2*p+1,1),(2*q+1,1),(2*r+1,0),(2*s+1,0)), h2[p,s,q,r])
        return ham
    

    def fci(self):
        ''' Exact solver
        '''
        [e,v] = scipy.sparse.linalg.eigsh(self._ham, 6, which='SA')
        for i in range(len(e)):
            S2 = v[:,i].conj().T.dot(self._s2.dot(v[:,i])).real
            N  = v[:,i].conj().T.dot(self._qubit_N.dot(v[:,i])).real
            E  = v[:,i].conj().T.dot(self._ham.dot(v[:,i])).real
            print("    State %4i: %12.8f au  <S2>: %12.8f (%6.1f) " %(i,e[i],S2,N))
            #print("      State %4i: %12.8f au  <N >: %12.8f " %(i,E,N))
        return e

    def print_scf_info(self):
        ''' Print SCF information
        '''
        Escf = self._hf.e_tot
        Enuc = self._Enuc
        nbasis = self._norb
        nk = self._nk
        nact = self._ncas
        noa = self._noa
        print('******************************************************')
        print('                   SCF Information                    ')
        print('                                                      ')
        print('          Number of basis sets      :%15i'   %nbasis   )
        print('          Number of active orbitals :%15i'   %nact     )
        print('          Number of active electron :%15i'   %noa      )
        print('          Number of k points        :  (%3i,%3i,%3i)' %(nk[0],nk[1],nk[2]))
        print('          SCF energy                :%15.8f' %Escf     )
        print('                                                      ')
        print('                    FCI Energy:                       ')
        self._ref_energy = self.fci()
        print('******************************************************')
        print('\n')

        
