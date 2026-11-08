from openfermion import FermionOperator, QubitOperator, hermitian_conjugated, normal_ordered
from openfermion.linalg import get_sparse_operator
from operators import Operators
from tools import mapping
import numpy as np
import copy

class HamiltonianOps(Operators):
    ''' Operators From Hamiltonian
    '''
    def __init__(self, scf, options):
        '''
        '''
        self._class = options['class'].upper()
        self._type  = options['exc_level'].upper()
        self._isym  = options['spin_sym'].upper()

        self._complex = options['complex']
        self._one_body = options['onebody']
        self._low_memory = options['low_memory']

        self._norb = scf._ncas
        self._nqubit = scf._nqubit
        self._noa = scf._noa
        self._nob = scf._nob
        self._nva = scf._nva
        self._nvb = scf._nvb
        self._ham = scf._qubit_ham
        #self._ham = mapping(scf._fermi_ham, scf._mapping, scf._nqubit, scf._noa*2)
        self._pbc = scf._name.lower()=='pbc_scf'
        if self._pbc:
            if scf._k2g:
                self._complex = False
            else:
                self._complex = True

        self._imatrix = False
        self._ops = []
        self._ops_mat = []
        self.kernel()

    def generate_pool(self):
        """
        """
        for ops, coeff in self._ham.terms.items():
            self.add_ops(ops,coeff)

        self._nops = len(self._ops)
        print(" Number of operators: ", self._nops)
        #for i in range(self.n_ops):
        #    print(i,self.qubit_ops[i])

        if not self._low_memory:
            self.get_matrix_operator()

        return 
    
    kernel = generate_pool

    def add_ops(self, op, coeff):
        '''
        '''
        nx = 0; ny = 0; not_x_and_y = False
        op_new = []; to_add = True
        for ti in op:
            if ti[1] == 'X':
                nx += 1 
                op_new.append(ti)
            elif ti[1] == 'Y':
                ny += 1 
                op_new.append(ti)
            else:
                not_x_and_y = True

        if (nx+ny) > 1: 
            op = copy.deepcopy(op_new)
            if op[0][1] == 'X': 
                op[0] = (op[0][0], 'Y') 
            else:
                op[0] = (op[0][0], 'X')
            op = QubitOperator(op, 1j)
 
            to_add = True
            nlen = len(self._ops)
            for i in range(nlen):
                if op==self._ops[i]:
                    to_add = False
                    break
             
            if op.many_body_order() > 0 and to_add:
                self._ops.append(op)


            if self._complex:
                op = copy.deepcopy(op_new)
                op = QubitOperator(op, 1j)

                to_add = True
                nlen = len(self._ops)
                for i in range(nlen):
                    if op==self._ops[i]:
                        to_add = False
                        break
                 
                if op.many_body_order() > 0 and to_add:
                    self._ops.append(op)


