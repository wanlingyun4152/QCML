import copy

from ansatz_pool import UCC, ADAPT, Custom
from fermion_operator import FermionOps
from qubit_operator import QubitOps
from hamiltonian_operator import HamiltonianOps
from scf_from_pyscf import pyscf_interface
from vqe_opt import optimizer, ioptimizer
from set_options import set_options
from reference_state import HFRefState, HFTaperState, CISRefState, QMFRefState, RefStateFromInput 
from tools import write_ansatz
from cnots_count import count_cnots


def run_vqe(X, options):
    ''' main driver for run VQE algorithms
    '''
  #  print('        ***************************          ')
  #  print('        *                         *          ')
  #  print('        *         VQEChem         *          ')
  #  print('        *                         *          ')
  #  print('        ***************************          ')

    options = set_options(options)

    Nop = max(options['ansatz']['Nu'],options['ansatz']['Nt'])
    op_class = options['ops']['class'].upper()
  #  print('')
  #  print(' Iterative Wave Function Ansatz: %s' %options['ansatz']['method'].upper())
  #  print(' Form of Transformation        : %s' %options['ansatz']['form'].upper())
    if op_class=='FERMIONIC':
        print(' Operator Pool                 : %s (%s, %s)' %(options['ops']['class'].upper(),
            options['ops']['exc_level'].upper(), options['ops']['spin_sym'].upper()))
    elif op_class=='QUBIT':
        print(' Operator Pool                 : %s (%s)' %(options['ops']['class'].upper(),
            options['ops']['spin_sym'].upper()))
    elif op_class=='HAMILTONIAN':
        print(' Operator Pool                 : %s' %(options['ops']['class'].upper()))
    else:
        print('Valid class operator pool: fermionic, qubit, hamiltonian')    
        exit()
  #  print(' Number of operators to update : %d (in each iteration)' %Nop)
  #  print('')

    method = options['vqe']['algorithm'].lower()
    if method=='fix-vqe':
        ansatz = fix_vqe(X, options)
    elif method=='adapt-vqe':
        ansatz = adapt_vqe(X, options)
    else:
        print('Incorrect vqe algorithms')
        exit()

    #if options['vqe']['save_ansatz']:
    #    write_ansatz(options, ansatz)

    if options['circuit']['count_cnot']:
        n_cnots = count_cnots(ansatz,options)
    return ansatz

def fix_vqe(X, options):
    ''' main driver for ucc vqe
    '''
    hf = pyscf_interface(X, options)

    ops_class = options['ops']['class'].lower()
    pool = options['ops']['ops_pool']
    if pool is None:
        if ops_class=='fermionic':
            pool = FermionOps(hf, options['ops'])
        elif ops_class=='qubit':
            pool = QubitOps(hf, options['ops'])
        else:
            print('Incorrect operator pool')
            exit()
        
    ref = reference(hf,options)

    method = options['ansatz']['method'].lower()
    if method=='ucc':
        ansatz = UCC(options['ansatz'], hf, pool, ref)
    elif method=='custom':
        ansatz = Custom(options['ansatz'], hf, pool, ref)
    else:
        print("Please check Ansatz method" %method)
        raise
  #  print("start bfgs")
    opt = optimizer(options)

    result = opt.run(ansatz)
    ansatz = result
    force = opt.get_force(ansatz, hf, X, options)
#    print("force:",force[2])
    print("force:",force)
    with open('force_true.txt','w') as out_file:
        for item in force:
            out_file.write(str(item.real)+'\t')
        out_file.write('\n')
        out_file.close()
    dipole = opt.get_dipole(hf,ansatz)
    print("dipole au:",dipole.real)
    print("dipole Debye:", 2.541746 * dipole.real)

    return ansatz


def adapt_vqe(X, options):
    ''' main driver for iterative variational quantum algorithms
        we need to prepare the reference state and operator pool
        for constructing wave function Ansatz.
    '''
    hf = pyscf_interface(X, options)

    ops_class = options['ops']['class'].lower()
    pool = options['ops']['ops_pool']
    if pool is None:
        if ops_class=='fermionic':
            pool = FermionOps(hf, options['ops'])
        elif ops_class=='qubit':
            pool = QubitOps(hf, options['ops'])
        elif ops_class=='hamiltonian':
            pool = HamiltonianOps(hf, options['ops'])
        else:
            print('Incorrect operator pool')
            exit()

    ref = reference(hf, options)

    ansatz = ADAPT(options['ansatz'], hf, pool, ref)

    opt = ioptimizer(options)

    ansatz = opt.run(ansatz)

    return ansatz
    

def reference(hf, options0):
    ''' reference state: 
            Hartree-Fock, Qubit Mean Field, or
            ADAPT wave function (ADAPT, IQEB, QUBIT)
    '''
    # default reference state: Hartree-Fock state
    options = copy.deepcopy(options0)
    itype = options['ref']['ref'].lower()
    if itype == 'hf':
        ref = HFRefState(hf)
    elif itype == 'hf_taper':
        ref = HFTaperState(hf)
    elif itype == 'qmf':
        ref = QMFRefState(hf, fix=options['ref']['fix'])
    elif itype == 'cis':
        ref = CISRefState(hf)
    else:
        options['opt']['maxiter'] = options['Nu']
        options['ansatz']['method'] = 'adaptu'
        options['ansatz']['Nu'] = 1
        options['ref']['ref'] = 'hf'
        if itype == 'adapt':
            options['ansatz']['name'] = 'adapt-vqe'
        elif itype == 'iqeb':
            options['ansatz']['name'] = 'iqeb-vqe'
        elif itype == 'qubit':
            options['ansatz']['name'] = 'qubit-vqe'
        else:
            print(' Please check input type of reference state')
            exit()
        ansatz = run_vqe(hf, options)
        ref = RefStateFromInput(hf,v=ansatz._state)

    return ref



























# backups
def iqeb_vqe(X, options):
    ''' Iterative Qubit Excitation based vqe
    '''
    hf = pyscf_interface(X, options)

    pool = QubitOps(hf, options)

    ref = reference(hf, options)

    ansatz = ADAPT(options['ansatz'], hf, pool, ref)

    opt = ioptimizer(options)

    ansatz = opt.run(ansatz)

    return ansatz


def qubit_vqe(X, options={'Nu':1}):
    ''' Qubit adapt-vqe
    '''
    return iqeb_vqe(X, options, 'qubit', 'qubit-vqe')


def rqubit_vqe(X, options={'Nu':1}):
    ''' Qubit adapt-vqe
    '''
    return iqeb_vqe(X, options, 'reduced_qubit', 'rqubit-vqe')


def ham_vqe(X, options):
    ''' Hamiltonian Qubit Operators
    ''' 
    hf = pyscf_interface(X, imap=options['scf']['mapping'])

    ref = reference(hf, options)

    pool = HamiltonianOps(hf)

    ansatz = ADAPT(options, hf, pool, ref)

    opt = ioptimizer(options)

    ansatz = opt.run(ansatz) 

    return ansatz


def hrsq_vqe(X, options={'Nu':1}):
    ''' Hamiltonian with removing symmetry qubits (hrsq)
    ''' 
    if not 'fix' in options: options['fix'] = False
    if not 'ref' in options: options['ref'] = 'qmf'

    hf = pyscf_interface(X, imap='SCBK')

    ref = reference(hf, options)

    pool = HamiltonianOps(hf)

    ansatz = ADAPT(options, 'hrsq-vqe', hf, pool, ref)

    opt = ioptimizer(options)

    ansatz = opt.run(ansatz) 

    return ansatz


def hrsq_tc_vqe(X, options={'Nu':1}):
    ''' Hamiltonian with removing two qubits
    ''' 
    if not 'imap' in options: options['imap'] = 'SCBK'
    hf = pyscf_interface(X, imap=options['imap'])

    if not 'fix' in options: options['fix'] = False
    ref = reference(hf,options)

    pool = HamiltonianOps(hf)

    ansatz = ADAPT(options, 'hrsq-vqe', hf, pool, ref)

    opt = ioptimizer(options)
    ansatz = opt.run(ansatz) 

    return ansatz
