

def set_options(options0):
    ''' Initialize input file

    vqe (vqe controls):
        'name'        : uccsd-vqe, uccgsd-vqe, qccsd-vqe, qccgsd-vqe,
                        adapt-vqe, iqeb-vqe, qubit-vqe, rqubit-vqe,
                        ham-vqe, hrsq
                        default : adapt-vqe 
        'algorithm'   : fix-vqe, adapt-vqe
                        default : adapt-vqe
        'save_ansatz' : default : false
        
    ops (operator pool classes):
        'class'       : fermionic, qubit, hamiltonian
        'exc_level'   : sd, gsd
        'spin_sym'    : general, sa, sc for fermionic operators
                        iqeb, qubit, reduced_qubit for qubit operators
        'complex'     : real for molecular system
                        complex for periodic system
        'onebody'     : if we include one body operators
        'sym_reduce'  : remove operators according to orbital symmetry
        'low_memory'  : if we store matrix form of operators in memory

    scf (PySCF class):
        'mapping'       : JW (Jordan-Wigner), BK (Bravyi-Kitaev),
                          SCBK (symmetry conserving Bravyi Kitaev)
        'ncas'          : number of active orbitals
        'ncore'         : number of core orbitals
        'mo_list'       : list of active orbitals
        'shift'         : penalty coefficient
        'check_mo_sign' : sign of molecular coefficients

    pbc (PySCF PBC class):
        'nk'            : k points
        'scaled_center' : scaled center for k-point sampling
        'k2g'           : Unit cell with k-point sampling to Gamma point supercell 

    ref (reference_state classes):
        'ref'         : HF (Hartree-Fock), QMF (Qubit Mean Field),
                        ADAPT, IQEB, QUBIT    
        'fix'         : if the reference state is optimized

    ansatz ( wave function Ansatz class):
        'method'      : ucc, adapt
        'form'        : The form of transformation (unitary, first-order 
                        Taylor expansion, transcorrelated correlation)
                        default : unitary
        'Nu'          : Number of Unitary operators to be updated
                        default : 1
        'Nt'          : Number of Taylor operators to be updated
                        default : 1
        'opt_all'     : if all parameters are optimized
                        default : True
        'nopt'        : number of iterations to be optimized
                        default : 1
        'gtol'        : tolerance of residual gradients
                        default : 1e-7
        'random'      : initialize parameters with random numbers
                        default : False
        'print_ham'   : print the number of terms in Hamiltonian
                        default : False
        'print_ansatz': print level 
                        default : -1

    opt ( optimizer class):
        'maxiter'     : maximum number of iterations of outer optimization
                        default : 300
        'tol'         : tolerance of convergence
                        default : 1e-3
        'cgiter'      : maximum number of iterations of inner optimization
                        default : 100
        'cgtol'       : tolerance of optimization algorithm
                        default : 1e-6
        'method'      : optimization algorithm (see scipy.optimize for details)
                        default : 'BFGS'
        'target_func' : the target function to be minimized,
                        default : 'VAR' (variance of energy)
        'print'       : print level
                        default : -1

    oo ( optimized orbital ):
        'basis'       : the reference basis set for split virtual orbital approach
                        default : 'minao'
        'low_level'   : low level approach to compute dynamic correlation
                        default : 'ccsd'
        'type'        : type of optimized orbital
                        default : 'hf'
        'diag'        : FCI solver for test
                        default : 'vqe'
        'ref_energy'  : if we compute the exact energy
                        default : true
    '''
    # default settings
    options =           {
                         'vqe'            : {},
                         'ops'            : {},
                         'scf'            : {},
                         'pbc'            : {},
                         'ref'            : {},
                         'ansatz'         : {},
                         'opt'            : {},
                         'oo'             : {},
                         'circuit'        : {}
                        }

    # options for VQE
    options['vqe'] =    {
                         'name'           : 'adapt-vqe',
                         'algorithm'      : 'adapt-vqe',
                         'save_ansatz'    : False
                        }

    # options for operator class
    options['ops'] =    {
                         'class'          : 'fermionic',
                         'exc_level'      : 'gsd',
                         'spin_sym'       : 'general', 
                         'sym_reduce'     : False,
                         'complex'        : False,
                         'onebody'        : True,
                         'low_memory'     : False,
                         'ops_pool'       : None
                        }

    # options for SCFFromPySCF class
    options['scf'] =    {
                         'mapping'        : 'JW',
                         'ncas'           : None,
                         'ncore'          : None,
                         'mo_list'        : None,
                         'sshift'         : 0,
                         'nshift'         : 0,
                         'taperoff'       : False,
                         'check_mo_sign'  : False,
                         'qmmm_coords'    : None,
                         'qmmm_charges'   : None
                        }

    # options for PBCSCFFromPySCF class
    options['pbc'] =    {
                        'nk'              : [1,1,1],
                        'scaled_center'   : [0,0,0],
                        'k2g'             : False,
                        'exxdiv'          : None
                        }

    # options for reference class
    options['ref']    = {
                         'ref'            : 'hf',
                         'fix'            : False,
                        }

    # options for wave function ansatz class
    options['ansatz'] = {
                         'method'         : 'adapt',
                         'form'           : 'unitary',
                         'Nu'             : 1,
                         'Nt'             : 1,
                         'opt_all'        : True,
                         'nopt'           : 1,
                         'gtol'           : 1e-7,
                         'random'         : False,
                         'print_ham'      : False,
                         'print_ansatz'   : False
                        }

    # options for optimization procedure
    options['opt'] =    {
                         'maxiter'        : 300,
                         'tol'            : 1e-3,
                         'niter_inner'    : 1,
                         'cgiter'         : 100,
                         'cgtol'          : 1e-6,
                         'method'         : 'BFGS',
                         'target_func'    : 'VAR',
                         'print'          : -1
                        }

    # options for split virtual orbitals
    options['oo'] =     {
                         'basis'          : 'minao',
                         'low_level'      : 'ccsd',
                         'type'           : 'hf',
                         'diag'           : 'vqe',
                         'ref_energy'     : True
                        }

    # options for quantum circuits
    options['circuit'] ={
                         'count_cnot'     : False
                        }

    # input
    for name in options0:
        for key,value in options0[name].items():
            options[name][key] = value

    options = personal_options(options)

    return options


def personal_options(options):
    # default settings for each abbreviation of algorithms 
    name = options['vqe']['name'].lower()
    if name=='uccsd-vqe' or name=='qccsd-vqe' or \
       name=='uccgsd-vqe' or name=='qccgsd-vqe':
        options['vqe']['algorithm'] = 'fix-vqe'
        options['ansatz']['method'] = 'ucc'
        options['ansatz']['form'] = 'unitary'
    
    if name=='uccsd-vqe':
        options['ops']['exc_level'] = 'sd'

    if name=='qccsd-vqe':
        options['ops']['exc_level'] = 'sd'
        options['ops']['class'] = 'qubit'
        options['ops']['form'] = 'iqeb'

    if name=='qccgsd-vqe':
        options['ops']['class'] = 'qubit'
        options['ops']['form'] = 'iqeb'

    if name=='iqeb-vqe':
        options['ops']['class'] = 'qubit'
        options['ops']['form'] = 'iqeb'

    if name=='qubit-vqe':
        options['ops']['class'] = 'qubit'
        options['ops']['form'] = 'qubit'
    
    if name=='rqubit-vqe':
        options['ops']['class'] = 'qubit'
        options['ops']['form'] = 'reduced_qubit'

    if name=='ham-vqe':
        options['ops']['class'] = 'hamiltonian'

    if name=='hrsq-vqe':
        options['ref']['ref'] = 'qmf'
        options['scf']['mapping'] = 'SCBK'
        options['ops']['class'] = 'hamiltonian'

    if name=='custom-vqe':
        options['vqe']['algorithm'] = 'fix-vqe'
        options['ansatz']['method'] = 'custom'

    return options
