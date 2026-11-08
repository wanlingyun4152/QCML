from scipy.optimize import minimize
import numpy as np
import datetime
from atimesb import expect_value
import openfermion
from tools import mapping

class optimizer:
    ''' VQE Optimization Driver
    '''
    def __init__(self, options=None):
        ''' 
        '''
        self._maxiter = options['opt']['maxiter']
        self._niter_inner = options['opt']['niter_inner']
        self._cgiter = options['opt']['cgiter']
        self._method = options['opt']['method']
        self._tol = options['opt']['tol']
        self._cgtol = options['opt']['cgtol']
        self._opt_func = options['opt']['target_func']
        self._print_level = options['opt']['print']
        self._print_ham = options['ansatz']['print_ham']


    def run(self, ansatz, params=None):
        ''' Run VQE Optimization Procedure
        '''
#        print("")
#        print(" ============================================= ")
#        print(" Iteration        Energy           Max Gradient    ")
        ansatz.update()

        pre_param = np.loadtxt("params.txt")
        pre_param = list(pre_param)
        self.pre_param = pre_param
        #    #params = list(np.random.uniform(
        #    #                - np.pi, np.pi, ansatz.get_nparams_opt()))
#        t1 = datetime.datetime.now()
        pre_energy = ansatz.energy(pre_param)
        #result = minimize(ansatz.energy,params,method=self._method,jac=ansatz.gradient,callback=ansatz.callback)
#        t2 = datetime.datetime.now()
        print("Predict energy : %16.10f" %pre_energy)
#        print("t1 :", t1)
#        print("t2 :", t2)
#        print("t2 - t1 :", t2 - t1, flush=True)

#        if ansatz._niter > 9:
#            print("\n %4i  %20.10f        %10.1e" %(ansatz._niter,ansatz._energy,ansatz._max_grad))
#        converged = self.check_converged(ansatz,result)
#        self.print_info(ansatz,result)

        return ansatz

    def get_force(self, ansatz, hf, X, options):
        from scf_from_pyscf import get_2e_integral
        from force import gradient_mo_operator
        h2 = get_2e_integral(hf)
        two_electron_integrals = np.asarray(h2.transpose(0, 2, 3, 1), order='C')
        atom_force_ops = gradient_mo_operator(hf, X, hf._mo_coeff, hf._h1, two_electron_integrals, with_pulay=True)
        #print(atom_force_ops)
        para = self.pre_param
        v = ansatz.state(para)
        force = []
        for i in range(len(atom_force_ops)):
            force.append(expect_value(v,v,atom_force_ops[i]))
        return force

    def get_sparse_dipole(self, dp, direction, hf):
        fermi_op = openfermion.FermionOperator()
        for p in range(dp.shape[1]):
            pa = 2*p
            pb = 2*p+1
            for q in range(dp.shape[2]):
                qa = 2*q
                qb = 2*q+1
                fermi_op += openfermion.FermionOperator(((pa,1),(qa,0)), dp[direction-1,p,q])
                fermi_op += openfermion.FermionOperator(((pb,1),(qb,0)), dp[direction-1,p,q])
        from tools import mapping
        qubit_op = mapping(fermi_op, hf._mapping, hf._nqubit, hf._noa*2)
        return(openfermion.linalg.get_sparse_operator(qubit_op))

    def get_dipole(self, hf, ansatz):
        dp = ansatz.scf._dipole
        #print("dp.shape:", dp.shape)
        #print("dp:", dp)
        direction = 3 #z
        dipole_ops = self.get_sparse_dipole(dp, direction, hf)
        #print("dipole_ops.shape:", dipole_ops.shape)
        #print("dipole_ops:", dipole_ops)
        v = ansatz.state(self.pre_param)
#        v = ansatz.ref._state
        dipole = expect_value(v,v,dipole_ops)
        return -dipole

    def check_converged(self, ansatz, result):
        ''' Check convergence of VQE optimizaiton procedure
        '''
        converged = False
        Energy = ansatz.energy(result['x'])
        if result['nit']==0:
            print(" Can't lower energy!")
            converged = True
        
        v = ansatz.state(result['x'])
        Hv = ansatz.scf._ham.dot(v)
        S = v.conj().T.dot(v)[0,0].real
        H2 = Hv.conj().T.dot(Hv)[0,0].real/S
        E2 = (Hv.conj().T.dot(v)[0,0].real)**2/S**2
        S2 = v.conj().T.dot(ansatz.scf._s2.dot(v))[0,0].real
        Hv = ansatz.scf._ham_bare.dot(v)
        Energy_bare = Hv.conj().T.dot(v)[0,0].real/S
        deltaH = np.sqrt(np.abs(H2-E2))
        Hv = ansatz.scf._ham_bare.dot(v)
        H2 = Hv.conj().T.dot(Hv)[0,0].real/S
        E2 = (Hv.conj().T.dot(v)[0,0].real)**2/S**2
        deltaH2 = np.sqrt(np.abs(H2-E2))

        error = 0
        if self._opt_func.upper() == 'VAR':
            if deltaH < self._tol:
                converged = True
            error = deltaH / self._tol
        elif self._opt_func.upper() == 'GMAX':
            if ansatz._max_res_grad < self._tol:
                converged = True
            error = ansatz._max_res_grad / self._tol
        elif self._opt_func.upper() == 'GNORM':
            if ansatz._norm_res_grad < self._tol:
                converged = True
            error = ansatz._norm_res_grad / self._tol
        else:
            print(' Target function to be optimized should be VAR or GRAD')
            exit()

      #  print(' delta Penalty H               = %12.8f' %deltaH)
      #  print(' delta H                       = %12.8f' %deltaH2)
      #  print(' max of residual gradient      = %12.8f' %ansatz._max_res_grad)
      #  print(' norm of residual gradient     = %12.8f' %ansatz._norm_res_grad)
      #  print(' Penalty Energy                = %12.8f' %Energy)
      #  print(' Energy                        = %12.8f' %Energy_bare)
      #  print(' S**2                          = %12.8f' %S2)
        
        return converged, error

    def print_info(self,ansatz,result):
        '''
        '''
        v = ansatz.state(result['x'])
        s2 = v.conj().T.dot(ansatz.scf._s2.dot(v))[0,0].real
        overlap = v.conj().T.dot(v)[0,0].real
        s2 = s2/overlap
        deltaE = ansatz._energy-ansatz.scf._ref_energy[0]
       
        #grad = ansatz.residual_gradient(result['x'],1)
      #  print(" ********************************************* ")
      #  print("                                               ")
      #  print(' %s converged with %8d (%8d opt) parameters' %(ansatz._name,ansatz.get_nparams(),ansatz.get_nparams_opt()))
      #  print(' Converged %s Energy: %16.10f  ||  <S2>: %8.6f' %(ansatz._name, ansatz._energy, s2))
      #  print(' Error of Energy (E-E_fci): %20.10e (%20.10e)' %(deltaE,deltaE*627.503) ) 
        if self._print_level > 1:
            print(result['x'])
            for i in range(len(ansatz._ops)):
                print(i,ansatz._ops[i])


class ioptimizer(optimizer):
     ''' Iterative VQE Optimization Driver
     '''
     def run(self, ansatz, params=None):
        ''' Run Iterative VQE Optimization Procedure
        '''
        if params is None: 
            if ansatz.get_nparams_opt() == 0:
                params = []
            else:
                params = ansatz._params
        params = np.asarray(params).flatten()

        options = {'gtol':self._cgtol, 'disp':False, 'maxiter':self._cgiter}
        # Iterative VQE procedure
        error = 100
        print('\n\n Begin Iterative VQE Optimization Procedure')
        for niter in range(self._maxiter):
            print(" ********************************************* ")
            print("              VQE Iteration %4d              " %niter)
            print(" ")
            params = ansatz.update(params, niter)
            if (niter+1)%self._niter_inner == 0 or error < 2.0:
                options['gtol'] = self._cgtol
            else:
                options['gtol'] = 1e-3
            print(" Iteration        Energy           Max Gradient    ")
            result = minimize(ansatz.energy, params, method=self._method, jac=ansatz.gradient,
                              options=options, callback=ansatz.callback)
            if ansatz._niter > 4:
                print("\n %4i  %20.10f        %10.1e" %(ansatz._niter,ansatz._energy,ansatz._max_grad))
            params = result['x']
            converged, error = self.check_converged(ansatz,result)
            if self._print_ham > 0:
                print(' Number of  Hamiltonian terms  = %12d'   %(ansatz.get_number_of_hamiltonian_terms(result['x'])))
            if converged:
                self.print_info(ansatz,result)
                break
            print(" ")
            print(" ")

        if not converged:
            print('Maximum number of iterations %4d is approached!!!' %self._maxiter)
            self.print_info(ansatz, result)

        return ansatz

