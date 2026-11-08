import scipy
import numpy
import copy

from functools import reduce

from pyscf import scf, mcscf, fci, ao2mo, symm, gto, qmmm
from pyscf.pbc import gto as pgto
from openfermion import FermionOperator, QubitOperator, normal_ordered
from openfermion.linalg import get_sparse_operator

from tools import s2_operator, mapping
from pbc_scf_from_pyscf import PySCF_PBC

from symmer import PauliwordOp
from symmer.projection import QubitTapering

def pyscf_interface(X, options):
    '''Interface to PySCF

    Args:
        X       : PySCF or Mole classes
        options : input data

    Returns: 
        SCF class

    '''
    if type(X) is PySCF or type(X) is PySCF_PBC:
        hf = X
    elif type(X) is gto.Mole:
        hf = PySCF(X, options)
        hf.run()
    elif type(X) is pgto.Cell:
        hf = PySCF_PBC(X, options)
        #hf.run()
    else:
        print('Input X should be SCFFromPySCF or Mole class')
        exit()

    return hf


def get_1e_integral(scf0, mo_core_list=None, mo_act_list=None, mo_coeff=None):
    '''CAS sapce one-electron hamiltonian

    Args:
        scf0 : a scfinit object

    Returns:
        A tuple, the first is the effective one-electron hamiltonian defined in CAS space,
        the second is the electronic energy from core.
    '''
    if mo_coeff is None: mo_coeff = scf0._mo_coeff
    if mo_core_list is None: mo_core_list = scf0._mo_core_list
    if mo_act_list is None: mo_act_list = scf0._mo_act_list
    mo_core = mo_coeff[:,mo_core_list]
    mo_cas = mo_coeff[:,mo_act_list]

    hcore = scf0._hf.get_hcore()
    energy_core = scf0._hf.energy_nuc()
    if mo_core.size == 0:
        corevhf = 0
    else:
        core_dm = numpy.dot(mo_core, mo_core.T) * 2
        corevhf = scf0._hf.get_veff(scf0._mol, core_dm)
        energy_core += numpy.einsum('ij,ji', core_dm, hcore)
        energy_core += numpy.einsum('ij,ji', core_dm, corevhf) * .5
    h1eff = reduce(numpy.dot, (mo_cas.T, hcore+corevhf, mo_cas))
    return h1eff, energy_core


def get_2e_integral(scf0, mo_act_list=None, mo_coeff=None):
    ''' CAS space two-electron integral

    Args:
        scf0 : a scfinit object

    Returns:
        two electron integral in MO
    '''
    if mo_coeff is None: mo_coeff = scf0._mo_coeff
    if mo_act_list is None: mo_act_list = scf0._mo_act_list
    mo_cas = mo_coeff[:,mo_act_list]

    eri_mo = ao2mo.full(scf0._mol, mo_cas)
    eri_mo = ao2mo.restore(1, eri_mo, scf0._ncas)
    return eri_mo


def get_1e_integral_qubit(scf0, mo_core_list=None, mo_act_list=None, mo_coeff=None):
    ''' CAS space one-electron integral 

    Args:
        scf0 : a scfinit object

    Returns:
        one electron integral in MO
    '''

    if mo_coeff is None: mo_coeff = scf0._mo_coeff
    if mo_core_list is None: mo_core_list = scf0._mo_core_list
    if mo_act_list is None: mo_act_list = scf0._mo_act_list
    h1,Enuc = get_1e_integral(scf0, mo_core_list, mo_act_list, mo_coeff)
    
    ncas = scf0._ncas
    h1q = numpy.zeros((2*ncas,2*ncas))
    for i in range(ncas):
        for j in range(ncas):
            h1q[2*i,2*j] = h1[i,j]
            h1q[2*i+1,2*j+1] = h1[i,j]
    return h1q 


def get_2e_integral_qubit(scf0, mo_act_list=None, mo_coeff=None):
    ''' CAS space two-electron integral

    Args:
        scf0 : a scfinit object

    Returns:
        two electron integral in MO
    '''
    if mo_coeff is None: mo_coeff = scf0._mo_coeff
    if mo_act_list is None: mo_act_list = scf0._mo_act_list
    mo_cas = mo_coeff[:,mo_act_list]
    eri = get_2e_integral(scf0, mo_coeff, nact, ncore) 
 
    ncas = scf0._ncas
    eriq = numpy.zeros((2*ncas,2*ncas,2*ncas,2*ncas))
    for p in range(ncas):
        for q in range(ncas):
            for r in range(ncas):
                for s in range(ncas):
                    eriq[2*p,2*q,2*r,2*s] = eri[p,q,r,s]
                    eriq[2*p+1,2*q+1,2*r,2*s] = eri[p,q,r,s]
                    eriq[2*p,2*q,2*r+1,2*s+1] = eri[p,q,r,s]
                    eriq[2*p+1,2*q+1,2*r+1,2*s+1] = eri[p,q,r,s]

    return eriq


class PySCF():
    ''' Prepare Hamiltonian for VQE using PySCF
    '''
    def __init__(self, mol, options):
        ''' Run Pyscf RHF 

        Args:
            mol:   Molecule object
            ncas:  number of active orbitals
            ncore: number of frozen occupied orbitals
        '''
        self._name = 'scf'
        # mapping from fermion to qubit
        self._mapping = options['scf']['mapping'].upper() 
        # orbital symmetry
        self._orbsym = None
        # product table of orbital symmetry
        self._prod_table = None
        # number of active orbitals
        ncas = options['scf']['ncas']
        # number of core orbitals
        ncore = options['scf']['ncore']
        # list of active orbitals
        mo_list = options['scf']['mo_list']
        # penalty coefficient for spin constraint
        self._sshift = options['scf']['sshift']
        # penalty coefficient for particle number constraint
        self._nshift = options['scf']['nshift']
        # Tapering off qubits
        self._taperoff = options['scf']['taperoff']

        # check the sign of molecular coefficients
        check_mo_sign = options['scf']['check_mo_sign']
        noa, nob = mol.nelec
        self._nbasis = mol.nao_nr()          # number of basis sets

        hf = scf.RHF(mol)
        qmmm_coords = options['scf']['qmmm_coords']
        qmmm_charges = options['scf']['qmmm_charges']
        if qmmm_charges:
            hf = qmmm.mm_charge(hf, qmmm_coords, qmmm_charges)
        hf.run()
        self._hf = hf
        self._mo_coeff = hf.mo_coeff
        if mol.symmetry:
            mo_sym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, hf.mo_coeff)
            mo_sym_new = ['1'+mo_sym[0]]
            for i in range(1,self._nbasis):
                mo_sym_i = mo_sym[i]
                n_order = 1
                for j in range(i):
                    if mo_sym_i==mo_sym[j]:
                        n_order += 1
                mo_sym_new.append(str(n_order)+mo_sym_i)
            print(mo_sym_new)
        else:
            mo_sym_new = []
            for i in range(0,self._nbasis):
                mo_sym_new.append(str(i+1)+'A') 

        if mo_list is None:
            if ncas is None: ncas = mol.nao_nr()
            if ncore is None: ncore = 0
            mo_list = [i for i in range(ncore,ncore+ncas)]
        elif isinstance(mo_list[0],str):
            mo_list_str = copy.deepcopy(mo_list)
            mo_list = []
            for mo in mo_list_str:
                for i in range(self._nbasis):
                    if mo==mo_sym_new[i]:
                        mo_list.append(i)
        
        self._mo_act_list = mo_list
        self._mo_core_list = []
        for i in range(0,noa):
            if i not in mo_list:
                self._mo_core_list.append(i)
 
        self._ncore = len(self._mo_core_list)
        self._ncas = len(mo_list)
        self._noa = noa - self._ncore        # number of active occupied orbitals (alpha)
        self._nob = nob - self._ncore        # number of active occupied orbitals (beta)
        self._nva = self._ncas - self._noa
        self._nvb = self._ncas - self._nob
        self._nqubit = self._ncas*2
        self._mol = mol

        if check_mo_sign:
            first_cal = False
            try:
                mo_coeff_old = numpy.load('mo.npy') 
            except:
                first_cal = True
            if not first_cal:
                S = mol.intor('int1e_ovlp')
                #print(numpy.einsum("ji,jk,kl->il",self._mo_coeff[:,mo_list],S,self._mo_coeff[:,mo_list]))
                ovlp = numpy.einsum("ji,jk,kl->il",mo_coeff_old,S,self._mo_coeff[:,mo_list])
                #print(ovlp)
                for i in range(len(mo_list)):
                    #print(numpy.sign(ovlp[i,i]))
                    self._mo_coeff[:,mo_list[i]] *= numpy.sign(ovlp[i,i])
#            numpy.save('mo.npy',self._mo_coeff[:,mo_list])

        self._nmo = self._mo_coeff.shape[1]
      #  print(hf.mo_energy)
        #taoyx homo-lumo
        mo_energy_ev = [i*27.2114 for i in hf.mo_energy]
      #  print("RHF mo_energy / eV :", mo_energy_ev)
        homo_index = self._ncore + max(self._noa, self._nob) -1
        lumo_index = self._ncore + max(self._noa, self._nob)
      #  print("HOMO Energy : %16.10f" %mo_energy_ev[homo_index], "eV")
      #  print("LUMO Energy : %16.10f" %mo_energy_ev[lumo_index], "eV")
        ###dipole
      #  hf.run()
        hf.dip_moment(unit='Debye')
        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        charge_center = numpy.einsum('i,ix->x', charges, coords)
      #  print("charge_center: ",charge_center)
        Cocc = hf.mo_coeff[:,self._mo_core_list]
        dm = (Cocc @ Cocc.T)*2
        with mol.with_common_orig((0,0,0)):
            ao_dip = mol.intor_symmetric('int1e_r', comp=3)
        el_dip = numpy.einsum('xij,ji->x', ao_dip, dm).real
      #  print("el_dip:", el_dip)
 
        mol_dip = charge_center - el_dip
      #  print("active-space charge_center: ",mol_dip)
        from pyscf import lib
        with mol.with_common_origin(mol_dip/mol.atom_charges().sum()):
            self._dipole = numpy.asarray([reduce(lib.dot, (self._mo_coeff.conj().T, x, self._mo_coeff))
                             for x in mol.intor_symmetric('int1e_r', comp=3)]) 
        ###dipole

        if mol.symmetry:
            self._orbsym = hf.orbsym[mo_list]
            self._prod_table = symm.direct_prod(
                numpy.arange(len(symm.symm_ops(mol.topgroup))),
                numpy.arange(len(symm.symm_ops(mol.topgroup))),
                mol.topgroup)
           # print(mol.topgroup,len(symm.symm_ops(mol.topgroup)))
           # print(self._orbsym)

    def run(self, h1=None, h2=None):
        '''
        '''
        if h1 is None: 
            self._h1, self._Enuc = get_1e_integral(self)
        else: 
            self._h1 = numpy.copy(h1)
       # print("Nuclear energy: %15.8f" %self._Enuc)

        self._fermi_s2, Sz = s2_operator(self._ncas)
        self._qubit_s2 = mapping(self._fermi_s2, self._mapping, self._nqubit, self._noa*2)
        self._s2 = get_sparse_operator(self._qubit_s2)

        self._fermi_ham_bare = self.hamiltonian(h1=h1, h2=h2)
        self._qubit_ham_bare = mapping(self._fermi_ham_bare, self._mapping, self._nqubit, self._noa*2)
        self._qubit_ham_bare.compress(1e-15)
        self._ham_bare = get_sparse_operator(self._qubit_ham_bare)

        self._fermi_ham = self._fermi_ham_bare
        if (abs(self._sshift) > 0):
            self._fermi_ham = self._fermi_ham + self._sshift/2 * self._fermi_s2

        nelec = self._noa + self._nob
        if (abs(self._nshift) > 0):
            self._fermi_ham = self._fermi_ham + self._nshift/2 * (self._fermi_N-nelec)*(self._fermi_N-nelec)

        self._qubit_ham = mapping(self._fermi_ham, self._mapping, self._nqubit, self._noa*2)
        nHq = len(self._qubit_ham.terms)
        self._qubit_ham.compress(1e-15)
        nHqc = len(self._qubit_ham.terms)
        print(f'The number of terms in the qubit Hamiltonin is compressed from {nHq} -> {nHqc}.\n')
        #self._ham = get_sparse_operator(self._qubit_ham)
        #self._ham = self._ham_bare + self._shift/2 * self._s2

        if self._taperoff:
            H = PauliwordOp.from_openfermion(self._qubit_ham)
            QT = QubitTapering(H)
           # print(f'Qubit tapering permits a reduction of {H.n_qubits} -> {H.n_qubits-QT.n_taper} qubits.\n')
           # print('The following symmetry generators were identified:\n')
           # print(QT.symmetry_generators); print()
           # print('which we may rotate onto the single-qubit Pauli operators\n')
           # print(QT.symmetry_generators.rotate_onto_single_qubit_paulis()); print()
           # print('via a sequence of Clifford operations R_k = e^{i pi/4 P_k} where:\n')
            for index, (P_k, angle) in enumerate(QT.symmetry_generators.stabilizer_rotations):
                P_k.sigfig=0
                print(f'P_{index} = {P_k}')
            self._qubit_ham = H.to_openfermion
            nHqc = len(self._qubit_ham.terms)
            print(f'The number of terms in the qubit-taper Hamiltonin is reduced from {nHq} -> {nHqc}.\n')

        self._ham = get_sparse_operator(self._qubit_ham)

        if self._mapping == 'SCBK':
            #self._ncas = self._ncas - 1
            self._nqubit = self._nqubit - 2

        self.print_scf_info()


    def hamiltonian_qubit(self, shift=0, h1=None, h2=None):
        ''' Build second quantazied Hamiltonian
            H = \sum_{pq} h1_{pq} a_p^+ a_q 
              + 1/2 \sum_{pqrs} h2_{psqr} a_p^+ a_q^+ a_r a_s
        '''
        nqubit = self._ncas*2
        ham = FermionOperator((),self._Enuc)
        if h1 is None: h1,Enuc = get_1e_integral_qubit(self)
        for p in range(nqubit):
            for q in range(nqubit):
                ham += FermionOperator(((p,1),(q,0)), h1[p,q])

        if h2 is None: h2 = get_2e_integral_qubit(self)
        for p in range(nqubit):
            for q in range(nqubit):
                for r in range(nqubit):
                    for s in range(nqubit):
                        ham += FermionOperator(((p,1),(q,1),(r,0),(s,0)), h2[p,s,q,r])

        if shift != 0: 
            ham = ham + shift * s2_operator(self._ncas)[0]

        return normal_ordered(ham)

    def hamiltonian(self, shift=0, h1=None, h2=None):
        ''' Build second quantazied Hamiltonian
            H = \sum_{pq} h1_{pq} a_p^+ a_q 
              + 1/2 \sum_{pqrs} h2_{psqr} a_p^+ a_q^+ a_r a_s
        '''
        nmo = self._ncas
        ham = FermionOperator((),self._Enuc)

        if h1 is None: h1 = numpy.copy(self._h1)
        for p in range(nmo):
            for q in range(nmo):
                ham += FermionOperator(((2*p,1),(2*q,0)), self._h1[p,q])
                ham += FermionOperator(((2*p+1,1),(2*q+1,0)), self._h1[p,q])

        if h2 is None: h2 = get_2e_integral(self)        
        for p in range(nmo):
            for q in range(nmo):
                for r in range(nmo):
                    for s in range(nmo):
                        ham += 0.5*FermionOperator(((2*p,1),(2*q,1),(2*r,0),(2*s,0)), h2[p,s,q,r])
                        ham += 0.5*FermionOperator(((2*p,1),(2*q+1,1),(2*r+1,0),(2*s,0)), h2[p,s,q,r])
                        ham += 0.5*FermionOperator(((2*p+1,1),(2*q,1),(2*r,0),(2*s+1,0)), h2[p,s,q,r])
                        ham += 0.5*FermionOperator(((2*p+1,1),(2*q+1,1),(2*r+1,0),(2*s+1,0)), h2[p,s,q,r])

        if shift != 0: 
            ham = ham + shift * s2_operator(nmo)[0]

        return normal_ordered(ham)

    def fci(self):
        '''
        '''
        #hamiltonian = self._ham + 0.5*self._s2
        #[e,v] = scipy.sparse.linalg.eigsh(hamiltonian, 3, which='SA')
        #for i in range(len(e)):
        #    S2 = v[:,i].conj().T.dot(self._s2.dot(v[:,i])).real
        #    print("    State %4i: %12.8f au  <S2>: %12.8f" %(i,e[i],S2))

        mc = mcscf.CASCI(self._hf, self._ncas, (self._noa,self._nob), ncore=self._ncore)
        mc.fcisolver = fci.addons.fix_spin_(fci.FCI(self._mol, self._mo_coeff), 0.5)
        mc.fcisolver.nroots = 6
        efci = mc.kernel()[0]
        #mc.analyze(ci=mc.ci[1],large_ci_tol=0.05, verbose=0)
        #print('   det-alpha,    det-beta,    CI coefficients')
        #for c,ia,ib in mc.fcisolver.large_ci(mc.ci, n_act, (n_a,n_b), tol=.05, return_strs=False):
        #        print('   %s       %s      %.12f' % (ia, ib, c))
        return efci

    def print_scf_info(self):
        Escf = self._hf.e_tot
        Enuc = self._Enuc 
        nbasis = self._nbasis
        nmo = self._ncas
        noa = self._noa
       # print('******************************************************')
       # print('                   SCF Information                    ')
       # print('                                                      ')
       # print('          Number of basis sets      :%15i'   %nbasis   )
       # print('          Number of active orbitals :%15i'   %nmo      )
       # print('          Number of active electron :%15i'   %noa      )
       # print('          SCF energy                :%15.8f' %Escf     )
       # print('                                                      ')
       # print('                    FCI Energy:                       ')
        if nmo*2 <= 16: self._ref_energy = self.fci()
       # print('******************************************************')
       # print('\n')
