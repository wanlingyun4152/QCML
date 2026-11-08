from openfermion import FermionOperator, hermitian_conjugated, normal_ordered 
from openfermion.linalg import get_sparse_operator
import numpy as np

class Operators:
    ''' Operator
    '''
    #def __init__(self, scf, itype, isym='general', icomplex=False, onebody=True, low_memory=False):
    def __init__(self, scf, options):
        '''
        '''
        self._class = options['class'].upper()
        self._type  = options['exc_level'].upper()
        self._isym  = options['spin_sym'].upper()
        self._complex = options['complex']
        self._one_body = options['onebody']
        self._sym_reduce = options['sym_reduce']
        self._low_memory = options['low_memory']

        self._norb = scf._ncas
        self._nqubit = scf._nqubit
        #print(f'nqubit : {self._nqubit}')
        self._noa = scf._noa
        self._nob = scf._nob
        self._nva = scf._nva
        self._nvb = scf._nvb
        self._orbsym = scf._orbsym
        self._prod_table = scf._prod_table
        self._pbc = scf._name.lower()=='pbc_scf'
        if self._pbc:
            self._kcons = scf._kcons
            self._m2k = scf._m2k
            if scf._k2g:
                self._complex = False
            else:
                self._complex = True

        self._imatrix = False
        self._ops = []
        self._ops_mat = []

        self.kernel()

    def get_matrix_operator(self):
        ''' Operator in Matrix Form
        '''
        if not self._imatrix:
            print(f' The number of qubits {self._nqubit}.')
            self._ops_mat = [None]*len(self._ops)
            print(" Generate Matrices Operators")
            for i in range(len(self._ops)):
                op = self._ops[i]
                self._ops_mat[i] = get_sparse_operator(op, n_qubits=self._nqubit)
            self._imatrix = True
        else:
            print(" Matrices Operators already exist!!!")
        return


    def ops_mat(self,i):
        ''' Get i-th Matrix Operator 
        '''
        if len(self._ops_mat)==0:
            op = get_sparse_operator(self._ops[i], n_qubits=self._nqubit)
        else:
            op = self._ops_mat[i]
        return op


    def get_index(self,i):
        ''' Get the Sequence of Index for Operator
        '''
        op = self._ops[i]
        opstring = ""
        for t in op.terms:
            opstring = "("
            for ti in t:
                opstring += str(int(ti[0]))
            opstring += ")"
        return opstring


    def k_conserve(self,ap_list,a_list):
        ''' Check momentum conservation for double excitation operators

            _kcons[p,q,r,s] : [k(p)-k(q)+k(r)-k(s)] \dot a = 2n \pi 
       
            operator in the form of a_p^+ a_q^+ a_r a_s
        '''
        if self._pbc and self._complex:
            if len(ap_list) == 1:
                kp = self._m2k[ap_list[0]][0]
                kq = self._m2k[a_list[0] ][0]
                conserved = (kp==kq)
            else:
                kp = self._m2k[ap_list[0]][0]
                kq = self._m2k[ap_list[1]][0]
                kr = self._m2k[a_list[0] ][0]
                ks = self._m2k[a_list[1] ][0]
                conserved = self._kcons[kr,kp,ks,kq]==1
        else:
            conserved = True

        return conserved


    def gradient(self,i,v,Hv,e0=None,v0=None):
        '''
        '''
        opA = self.ops_mat(i)
        E = Hv.conj().T.dot(v)[0,0]
        S = v.conj().T.dot(v)[0,0]
        #Hv = (Hv*S-v*E)/(S*S)
        Grad = 2*(Hv.conj().T.dot(opA.dot(v)))[0,0]/S
        #if e0 != None:
        #    S = v.conj().T.dot(v0)
        #    Grad -= (2*e0*S)*(v0.conj().T.dot(opA.dot(v)))[0,0]
        Grad = Grad.real

        return Grad

    def kernel(self):
        if self._type=='SD':
            self.sd_ops_pools()
        elif self._type=='GSD':
            self.gsd_ops_pools()
        else:
            print('Incorrect type of operator pool')
            exit()
        print(" Number of Operator: %6d" %len(self._ops))

    def sd_ops_pools(self):
        ''' single and double excitation operators
        '''
        noa = self._noa; nva = self._nva
        for i in range(0,noa):
            for a in range(noa,noa+nva):
                self.one_body_operator(a,i)

        for i in range(0,noa):
            for j in range(i,noa):
                for a in range(noa,noa+nva):
                    for b in range(a,noa+nva):
                        self.two_body_operator(a,b,i,j)

        if not self._low_memory:
            self.get_matrix_operator()

    def gsd_ops_pools(self):
        ''' general single and double excitation operators
        '''
        norb = self._norb
        for p in range(0,norb):
            for q in range(p,norb):
                if self._one_body and self.to_add([p],[q]):
                    self.one_body_operator(p,q)

        pq = 0
        for p in range(0,norb):
            for q in range(p,norb):
                pq += 1; rs = 0
                for r in range(0,norb):
                    for s in range(r,norb):
                        rs += 1
                        if (pq > rs or not self.to_add([p,q],[r,s])):
                            continue
                        self.two_body_operator(p,q,r,s)

        #for i in range(len(self._ops)):
        #    print(self._ops[i])

        if not self._low_memory:
            self.get_matrix_operator()

    def two_identical_index(self,i,j,k,l):
        ''' remove 4-index operators with any two index being equal
        '''
        is_two_identical_index = False
        if i==j or i==k or i==l or j==k or j==l or k==l:
            is_two_identical_index = True
        #return False
        return is_two_identical_index

    def check_symmetry_conversing_excitations(self,occ_indices_spatial: list,
                                               vir_indices_spatial: list):
        """
        Assume that occ_indices_spin are taken from doubly occupied spatial
        orbitals, and vir_indices_spin are taken from zero occupied spatial
        orbitals.
        """
        if not self._sym_reduce:
            return True

        if self._orbsym is None or self._prod_table is None:
            return True

        unique_orbsym = list(np.unique(self._orbsym))
        t = []
        all_indices = occ_indices_spatial + vir_indices_spatial
        n_indices = len(all_indices)
        symm_result = 0
        for i in range(n_indices):
            symm_result = self._prod_table[symm_result][
                unique_orbsym.index(self._orbsym[all_indices[i]])]
        print(occ_indices_spatial,vir_indices_spatial,symm_result)

        if symm_result == 0:
            return True

        return False


    def to_add(self, ap_list:list, a_list:list):
        '''
        Args: 
            ap_list : creation operator lists
            a_list  : annihilation operator lists
        '''
        symmetry_ok = self.check_symmetry_conversing_excitations(ap_list,a_list)
        k_conserved = self.k_conserve(ap_list,a_list)
        return (symmetry_ok and k_conserved)
