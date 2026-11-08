import scipy
#import operator_pools
#import pyscf_helper 
#import circuit_operator_pools
import rdm

#from qiskit.aqua.operators import WeightedPauliOperator as op
from qiskit import transpile, Aer
#from qiskit.providers.aer import Aer
from qiskit.quantum_info import Pauli
#from qiskit import QuantumCircuit, QuantumRegister
import re
import random

import pyscf
from pyscf import lib
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, cc
from pyscf.cc import ccsd

import openfermion 
import projectq
from openfermion import *
#from tVQE import *
import numpy as np
    
def _Q_dagger(p: int):
    Xp = openfermion.QubitOperator(((p, "X"), ), 1.)
    Yp = openfermion.QubitOperator(((p, "Y"), ), 1.)
    result = 0.5 * (Xp - 1.j * Yp)
    return result


def _Q(p: int):
    Xp = openfermion.QubitOperator(((p, "X"), ), 1.)
    Yp = openfermion.QubitOperator(((p, "Y"), ), 1.)
    result = 0.5 * (Xp + 1.j * Yp)
    return result

# Convert openfermion.QubitOperator to projectq.ops.QubitOperator
def of_qubitOp_to_pq_qubitOp(of_qubitOp: openfermion.QubitOperator, real_coeffs: bool=False):
    pq_qubitOp = projectq.ops.QubitOperator()
    for qubitOp in of_qubitOp.terms:
        pq_qubitOp += projectq.ops.QubitOperator(qubitOp, of_qubitOp.terms[qubitOp].imag * -1. if real_coeffs else of_qubitOp.terms[qubitOp])
    return pq_qubitOp

# return the gates for constructing the exp(qubitOp_pq) circuit
def pq_expm_trotter_circuit_single(qubitOp_pq: projectq.ops._qubit_operator.QubitOperator):
    """
    Circuit example for exp(qubitOp_pq):
                 ┌───────────┐   ┌───────────┐        ┌───────────┐   ┌───────────┐   
        q0_0: ───┤           ├───┤           ├────────┤           ├───┤           ├───
                 │           │   │           │        │           │   │           │   
        q0_1: ───┤           ├───┤           ├────────┤           ├───┤           ├───
                 │ gate_list │   │ CNOT_list │        │ CNOT_list │   │ gate_list │   
        q0_2: ───┤           ├───┤           ├────────┤           ├───┤           ├───
                 │           |   │           | ┌────┐ │           |   │           |   
        q0_3: ───┤           ├───┤           ├─┤ Rz ├─┤           ├───┤           ├───
                 └───────────┘   └───────────┘ └────┘ └───────────┘   └───────────┘   
    
    Number of CNOTS is 2 * len(CNOT_list)
    """
    assert(len(list(qubitOp_pq.terms.items())) == 1)
    assert(np.isclose(list(qubitOp_pq.terms.values())[0].real, 0.))
    coeff = list(qubitOp_pq.terms.values())[0]
    t = coeff / 1.j
    gate_dict = {
        "X": projectq.ops.H,
        "Y": projectq.ops.MatrixGate(2**0.5 / 2. * np.array([[1.0, -1.j], 
                                                             [1.j, -1.0]])),
    }
    P_tuple = list(qubitOp_pq.terms.keys())[0]
    gate_list = []
    CNOT_list = []
    for i in P_tuple:
        qubit_idx = i[0]
        gate_symbol = i[1]
        if (gate_symbol in gate_dict):
            gate_list.append((gate_dict[gate_symbol], qubit_idx))
    for i in range(len(P_tuple) - 1):
        ctrl_qubit_idx = P_tuple[i + 1][0]
        op_qubit_idx = P_tuple[i][0]
        CNOT_list.append((projectq.ops.CNOT, (ctrl_qubit_idx, op_qubit_idx)))
    return gate_list, CNOT_list


def projectq_count_cnots(ansatz: list, nqubits: int=0, ifermion: bool=True, IQEB: bool=False):
    """
    ansatz[0]: Any
    ansatz[1]: list of sparse matrices of the QubitOperators
    ansatz[2]: list of FermionOperators

    Variational parameters are not important since we only want to calculate the number of CNOTs
    """
    
    # ansatz[2] is a list of FermionOperator. Need to extract indices of ansatz[2]
    # and construct the corresponding qubit excitation operators.
    ansatz_qubitOp = []
    if ifermion:
        for i in range(len(ansatz[2])):
           tmp_op = ansatz[2][i] # its type is FermionOperator
           qubitOp_i = openfermion.jordan_wigner(tmp_op)
           ansatz_qubitOp.append(qubitOp_i)
    else:
        for i in range(len(ansatz[2])):
            tmp_op = ansatz[2][i] # its type is FermionOperator
            if IQEB:
                qubitOp_i = openfermion.QubitOperator()
                for term, coeff in tmp_op.terms.items():
                    # i is something like ((2, 1), (3, 1), (1, 0), (0, 0))
                    qubitOp_ii = openfermion.QubitOperator((), 1.)
                    for one_term in term:
                        orb_idx = one_term[0]
                        excitation = (one_term[1] == 1)
                        if excitation is True:
                            qubitOp_ii *= _Q_dagger(orb_idx)
                        else:
                            qubitOp_ii *= _Q(orb_idx)
                    qubitOp_i += coeff * qubitOp_ii
            else:
                qubitOp_i = tmp_op
            ansatz_qubitOp.append(qubitOp_i)
   
    # Check result
    for i in range(len(ansatz_qubitOp)):
        ax = np.linalg.norm(
            (openfermion.get_sparse_operator(
                ansatz_qubitOp[i], nqubits)-ansatz[1][i]).todense())
        bx = np.linalg.norm(
            (openfermion.get_sparse_operator(
                -ansatz_qubitOp[i], nqubits)-ansatz[1][i]).todense())
        assert(np.isclose(min(np.abs(ax),np.abs(bx)),0.0))

    ansatz_qubitOp_sum = openfermion.QubitOperator()
    for i in ansatz_qubitOp:
        ansatz_qubitOp_sum += i
    ansatz_qubitOp_for_circuit = list(ansatz_qubitOp_sum)

    ansatz_qubitOp_for_circuit_pq = [of_qubitOp_to_pq_qubitOp(i) for i in ansatz_qubitOp_for_circuit]
    n_CNOTS = 0
    for i in ansatz_qubitOp_for_circuit_pq:
        gate_list_i, CNOT_list_i = pq_expm_trotter_circuit_single(i)
        n_CNOTS += 2 * len(CNOT_list_i)

    return n_CNOTS

def fermi2qiskit(A0,nqubits,ifermion,IQEB):
    if ifermion:
        A = openfermion.transforms.jordan_wigner(A0)
    else:
        A = openfermion.QubitOperator()
        if IQEB:
            for term, coeff in A0.terms.items():
                # i is something like ((2, 1), (3, 1), (1, 0), (0, 0))
                qubitOp_ii = openfermion.QubitOperator((), 1.)
                for one_term in term:
                    orb_idx = one_term[0]
                    excitation = (one_term[1] == 1)
                    if excitation is True:
                        qubitOp_ii *= _Q_dagger(orb_idx)
                    else:
                        qubitOp_ii *= _Q(orb_idx)
                A += coeff * qubitOp_ii
        else:
            A = A0

    sum = []
    print("wanly:",A)
    for term in A.terms:
        zz = 0
        coeff = np.imag(A.terms[term])
        if coeff > 0:
            sign = '+'
        if coeff < 0:
            sign = '-'
        term = str(term)
        Bin = np.zeros((2 * nqubits,), dtype=int)
        X_pat = re.compile("(\d{,2}), 'X'")
        X = X_pat.findall(term)
        if X:
            for i in X:
                i = int(i)
                Bin[nqubits + i] = 1
        Y_pat = re.compile("(\d{,2}), 'Y'")
        Y = Y_pat.findall(term)
        if Y:
            for i in Y:
                i = int(i)
                Bin[nqubits + i] = 1
                Bin[i] = 1
        Z_pat = re.compile("(\d{,2}), 'Z'")
        Z = Z_pat.findall(term)
        if Z:
            for i in Z:
                i= int(i)
                Bin[i] = 1
        string = ''
        for k in range(nqubits):
            if Bin[k] == 0:
                if Bin[k + nqubits] == 1:
                    string += 'X'
                if Bin[k + nqubits] == 0:
                    string += 'I'
            if Bin[k] == 1:
                if Bin[k + nqubits] == 1:
                    string += 'Y'
                if Bin[k + nqubits] == 0:
                    string += 'Z'
        print("wanly:",string)
        p = Pauli(string)
        sum.append([-1*random.random(), p])
        print(sum)
    return op(paulis=sum)


def count_cnots(ansatz, options):
    '''
    '''
    ops_list = []
    print(ansatz._params_group)
    for i in range(len(ansatz._ops)):
        if ansatz._params_group[i][2].lower()=='adapt':
            ops = ansatz._ops[i]
            for j in ops:
                ops_list.append(ansatz.pool._ops[j])
        if ansatz._params_group[i][2].lower()=='ucc':
            ops = ansatz._ops[i]
            print("ops:",ops)
            for j in ops:
                ops_list.append(ansatz.pool._ops[j])
    print("ops_list",ops_list)
    op_type = options['ops']['class'].lower()
    op_symm = options['ops']['spin_sym'].lower()
    ifermion = False; IQEB = False
    if op_type=='fermionic': ifermion = True
    if op_type=='qubit' and op_symm=='iqeb': IQEB = True
    n_cnots = qiskit_count_cnots(ops_list, ansatz.scf._nqubit, ifermion, IQEB)
    #print(ops_list)

    print('VQE include %12d two-qubit gates' %n_cnots)
  
    return n_cnots


def qiskit_count_cnots(ops_list: list, nqubits: int=0, ifermion: bool=True, IQEB: bool=False):
    circuit_list = []
    #q = QuantumRegister(nqubits, name='q')
    #qc = QuantumCircuit(nqubits)
    n_cnots = 0

    print("len(ops_list)", len(ops_list))
    for i in range(len(ops_list)):
       tmp_op = ops_list[i] # its type is FermionOperator
       A = fermi2qiskit(tmp_op,nqubits,ifermion,IQEB)
       A_circuit = A.evolve(state_in=None,
                 evo_time=1,
                 num_time_slices=1,
                 #quantum_registers=q,
                 expansion_mode='trotter',
                 expansion_order=1)
       A_final_circuit = transpile(A_circuit,
                           basis_gates=['u3', 'cx', 'id'],
                           optimization_level=3,
                           backend=Aer.get_backend('statevector_simulator')
                           )
       #print(A_final_circuit.count_ops())
       n_cnots += A_final_circuit.count_ops()['cx']

    return n_cnots
   
def qeb_count_cnots(ansatz, options):
    ''' 
    '''
    op_type = options['ops']['class'].lower()
    op_symm = options['ops']['spin_sym'].lower()
    if op_type=='fermionic' and op_symm=='general':
        itype = 1
    elif op_type=='qubit' and op_symm=='iqeb':
        itype = 2
    elif op_type=='qubit' and op_symm=='iqeb':
        itype = 3
    else:
        print('Type of operators is not supported')
        exit()

    n_cnots = 0
    for op in range(ansatz._ops):
        n_cnots += cnot_in_op(op, itype)
    
    return n_cnots

def cnot_in_op(op, itype):
    '''
    '''
    maxL = 0
    for t in op:
        maxL = max(maxL, len(t))
 
    n = 0
    if maxL==2:
       if itype==1: n=2*abs(k-i)+1
       elif itype==2: n=2
       elif itype==3: n=2
    elif maxL==4:
       if itype==1: n=2*abs(l+j-i-k)+9
       elif itype==2: n=13
       elif itype==3: n=6
    else:
       print('3-index operator')
       exit()

    return n


        
