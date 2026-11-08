from fermion_operator import FermionOps
from openfermion import FermionOperator
from atimesb import expm_multiply, expect_value, csr_matvec_a
from openfermion import QubitOperator, jordan_wigner, hermitian_conjugated
from tools import csc_zeros, csc_identity
import numpy as np
import math
import copy
import sys
sys.setrecursionlimit(10000)

class Ansatz:
    ''' Wave Function Ansatz
    '''
    def __init__(self, options, scf, pool, ref):
        '''
        '''
        # name of wave function Ansatz
        self._name = options['method']
       
        # Initialize scf, pool, and reference classes
        self.scf  = scf
        self.pool = pool
        self.ref  = ref

        self._pauliop = ((pool._class=='HAMILTONIAN') or
                         (pool._class=='QUBIT' and pool._isym=='QUBIT'))

        # save energy and state
        self._energy = None
        self._state  = None

        # Initialize reference state's parameters
        # HF: None; QMF: (theta,psi)
        self._params = [ref._params] 
        self._ops    = [[]]

        # _params_group = [[ nops, fixed, unitary],...]
        # nops: number of operators; 
        # fixed: parameters to be optimized or not (True: fixed; False: opt)
        # unitary: exponential type e^(tau_i) or first-order Taylor expansion (1+tau_i)
        self._params_group  = self.ref._params_group 
        # _ref_fix: QMF parameters to be optimized or not
        self._ref_fix       = self.ref._params_group[0][1]
        # _normalized_wf: wave function is normalizd or not
        self._normalized_wf = True

        # save input control data
        self._niter         = 0
        self._options       = options
        self._gtol_res      = options['gtol']
        self._print_level   = options['print_ansatz']


    def energy(self, parameters):
        ''' E = <v|H|v> / <v|v>
        '''
        v = self.state(parameters)
        #E = v.conj().T.dot(self.scf._ham.dot(v))[0,0]
        E = expect_value(v,v,self.scf._ham)
        # if the wave function is not normalized
        if not self._normalized_wf:
            E = E/v.conj().T.dot(v)[0,0]
        assert(np.isclose(E.imag,0))
        self._energy = E.real
        return E.real


    def state(self, parameters):
        ''' |v> = Un ... U2 U1 |ref>
        '''
        params= self.parse_params(parameters)

        #  fixed = _params_group[0][1]
        if self._params_group[0][1]:
            v = self.ref._state
        else:
            v = self.ref.state(params[0])

        for i in range(1, len(params)):
            v = self.transform(v, self._ops[i], params[i], self._params_group[i][2])

        self._state = v
        return v


    def get_HSv(self, parameters):
        '''
        '''
        bra = []; ket = []; overlap = []; 
        v = self.state(parameters)
        #Hv = self.scf._ham.dot(v)
        Hv = csr_matvec_a(self.scf._ham, v)

        params= self.parse_params(parameters)
        bra.insert(0,Hv)
        if not self._normalized_wf: overlap.insert(0,v)
        for i in reversed(range(1, len(params))):
            params0 = (-1.0)*np.asarray(params[i])
            Hv = self.transform(Hv, self._ops[i], params0, self._params_group[i][2], True)
            bra.insert(0,Hv)
            if not self._normalized_wf:
                v = self.transform(v, self._ops[i], params0, self._params_group[i][2], True)
                overlap.insert(0,v)
        
        if self._params_group[0][1]:
            v0 = self.ref._state
        else:
            v0 = self.ref.state(params[0])

        ket.append(v0)
        for i in range(1, len(params)):
            v0 = self.transform(v0, self._ops[i], params[i], self._params_group[i][2])
            ket.append(v0)
         
        return bra, ket, overlap


    def get_Hv(self, parameters, idx, it=0, ovlp=False):
        '''
        '''
        v = self.state(parameters)
        if ovlp:
            Hv = copy.deepcopy(v)
        else:
            #Hv = self.scf._ham.dot(v)
            Hv = csr_matvec_a(self.scf._ham, v)

        params= self.parse_params(parameters)
        for i in reversed(range(idx+1, len(params))):
            params0 = (-1.0)*np.asarray(params[i])
            Hv = self.transform(Hv, self._ops[i], params0, self._params_group[i][2], True)
        
        if self._params_group[0][1]:
            v = self.ref._state
        else:
            v = self.ref.state(params[0])

        for i in range(1, min(idx+1-it,len(params))):
            v = self.transform(v, self._ops[i], params[i], self._params_group[i][2])
         
        return Hv, v


    def parse_params(self, parameters):
        ''' Fold 1-dimensional Array into 2-dimensional Array
        '''
        ioff = 0; params = self._params
        for i in range(0,len(self._params_group)):
            itype = self._params_group[i]
            if not itype[1]:
                params[i] = parameters[ioff:ioff+itype[0]]
                ioff += itype[0]

        #print(ioff,len(parameters))
        assert(ioff==len(parameters))       
        return params


    def transform(self, v, ops, params, method, ireversed=False):
        '''
        '''
        method = method.upper()
        if method == 'UCC':
            T = csc_zeros(self.scf._nqubit)
            for i in range(0, len(params)):
                T += params[i]*self.pool.ops_mat(ops[i])
            v = expm_multiply(T, v)
        elif method == 'TAYLOR':
            T = csc_identity(self.scf._nqubit)
            for i in range(0, len(params)):
                T += params[i]*self.pool.ops_mat(ops[i])
            #v = T.dot(v) 
            v = csr_matvec_a(T,v)
        elif method == 'ADAPT':
            if ireversed:
                for i in reversed(range(0, len(params))):
                    if self._pauliop:
                        v = (math.cos(params[i])*v + 
                             math.sin(params[i])*csr_matvec_a(self.pool.ops_mat(ops[i]),v))
                    else:
                        v = expm_multiply(params[i]*self.pool.ops_mat(ops[i]), v)
            else:
                for i in range(0, len(params)):
                    if self._pauliop:
                        v = (math.cos(params[i])*v + 
                             math.sin(params[i])*csr_matvec_a(self.pool.ops_mat(ops[i]),v))
                    else:
                        v = expm_multiply(params[i]*self.pool.ops_mat(ops[i]), v)
        else:
            print("Transformation Form %s!!! Trying UCC, Taylor and ADAPT" %method)
            exit()

        return v

  
    def get_nparams_opt(self):
        '''
        '''
        nparams = 0
        for i in range(0, len(self._params_group)):
            if not self._params_group[i][1]:
                nparams += self._params_group[i][0]
        return nparams


    def get_nparams(self):
        '''
        '''
        nparams = 0
        for i in range(0, len(self._params_group)):
            nparams += self._params_group[i][0]
        return nparams


    def gradient_ref(self, parameters, Hv=None):
        '''
        '''
        Grad = []
        params = self.parse_params(parameters)
        if not self._ref_fix:
            #v = self.state(parameters)
            #if ovlp:
            #    Hv = copy.deepcopy(v)
            #else:
            #    Hv = self.scf._ham.dot(v)
        
            #for i in reversed(range(1, len(params))):
            #    params0 = (-1.0)*np.asarray(params[i])
            #    Hv = self.transform(Hv, self._ops[i], params0, self._params_group[i][2], True)

            Grad = self.ref.gradient(params[0], Hv.conj().T)

        return Grad


    def gradient_ucc(self,parameters,ibeg,iend):
        '''
        '''
        Grad = []
        _eps = 1e-8
        for i in range(ibeg,iend):
            params1 = copy.deepcopy(parameters)
            params1[i] = params1[i] + _eps
            e1 = self.energy(params1)
            params1[i] = params1[i] - _eps*2
            e2 = self.energy(params1)
            Grad.append((e1-e2)/(_eps*2))

        Grad = np.asarray(Grad)
        return Grad


    def gradient_taylor(self, ops, params, Hv, v0):
        '''
        '''
        grad = []
        for i in range(len(params)):
            opA = self.pool.ops_mat(ops[i])
            grad.append(2*expect_value(Hv,v0,opA).real)
            #grad.append(2*Hv.conj().T.dot(opA.dot(v0)).toarray()[0][0].real)
        return np.asarray(grad)


    def gradient_adapt(self, ops, params, Hv, v0):
        '''
        '''
        i = len(params); grad = []
        grad = self.recurse(ops, params, grad, Hv, v0, i)
        return grad


    def recurse(self, ops, params, grad, Hv, v0, term):
        '''
        '''
        if term == len(params):
            Hv = Hv
            v0 = v0
        else:
            opA = self.pool.ops_mat(ops[term])
            if self._pauliop:
                Hv = (math.cos(params[term])*Hv - 
                      math.sin(params[term])*csr_matvec_a(opA,Hv))
                v0 = (math.cos(params[term])*v0 - 
                      math.sin(params[term])*csr_matvec_a(opA,v0))
            else:
                Hv = expm_multiply(-opA*params[term], Hv)
                v0 = expm_multiply(-opA*params[term], v0)

        opA = self.pool.ops_mat(ops[term-1])
        grad.insert(0,(2*expect_value(Hv,v0,opA).real))
        #grad.insert(0,(2*Hv.conj().T.dot(opA.dot(v0)).toarray()[0][0].real))
        if term > 1:
            term -= 1
            self.recurse(ops, params, grad, Hv, v0, term)
        return np.asarray(grad)


    def gradient2(self,parameters):
        '''
        '''
        Grad0 = []; ioff = 0
        E = self.energy(parameters)
        v = self.state(parameters)
        S = v.conj().T.dot(v)[0,0]
        assert(np.isclose(S.imag,0))
        S = S.real
        params = self.parse_params(parameters)
        for i in range(len(params)):
            if not self._params_group[i][1]:
                method = self._params_group[i][2].upper()
                nparams = self._params_group[i][0]
                if method == 'REF':
                    Grad = self.gradient_ref(parameters)
                    if not self._normalized_wf:
                        Ovlp = self.gradient_ref(parameters,True)
                elif method == 'UCC':
                    Grad = self.gradient_ucc(parameters,ioff,ioff+nparams)
                elif method == 'ADAPT':
                    Hv, v = self.get_Hv(parameters, i, 0)
                    Grad = self.gradient_adapt(self._ops[i],params[i],Hv,v)
                    if not self._normalized_wf:
                        v0, v = self.get_Hv(parameters, i, 0, True)
                        Ovlp = self.gradient_adapt(self._ops[i],params[i],v0,v)
                elif method == 'TAYLOR':
                    Hv, v = self.get_Hv(parameters, i, 1)
                    Grad = self.gradient_taylor(self._ops[i],params[i],Hv,v)
                    if not self._normalized_wf:
                        v0, v = self.get_Hv(parameters, i, 1, True)
                        Ovlp = self.gradient_taylor(self._ops[i],params[i],v0,v)
                else:
                    print('Invalid Transformation Type')
                    exit()
                ioff += nparams

                for i in range(len(Grad)):
                    if not self._normalized_wf:
                        Grad[i] = (Grad[i]-E*Ovlp[i])/S
                    Grad0.append(Grad[i])

        #print(Grad0)
        #Grad0 = self.gradient_finite_difference(parameters)
        #print(Grad0)

        Grad = np.asarray(Grad0)
        self._max_grad = np.max(np.abs(Grad))

        return np.asarray(Grad)


    def ops_type(self):
        '''
        '''
        exist_ADAPT = False; exist_TAYLOR = False
        for i in range(1,len(self._ops)):
            itype = self._params_group[i][2].upper() 
            if itype == 'ADAPT':
                exist_ADAPT = True
            elif itype == 'TAYLOR':
                exist_TAYLOR = True

        if exist_ADAPT and exist_TAYLOR:
            itype = 'HYBRID'
        elif exist_ADAPT:
            itype = 'ADAPT'
        elif exist_TAYLOR:
            itype = 'TAYLOR'
        else:
            itype = 'OTHER'

        return itype

    def gradient(self,parameters):
        '''
        '''
        Grad0 = []; ioff = 0
        E = self.energy(parameters)
        v = self.state(parameters)
        S = v.conj().T.dot(v)[0,0]
        assert(np.isclose(S.imag,0))
        S = S.real

        bra, ket, v0 = self.get_HSv(parameters)

        params = self.parse_params(parameters)
        for i in range(0,len(params)):
            if not self._params_group[i][1]:
                method = self._params_group[i][2].upper()
                nparams = self._params_group[i][0]
                if method == 'REF':
                    Grad = self.gradient_ref(parameters, bra[0])
                    if not self._normalized_wf:
                        Ovlp = self.gradient_ref(parameters, v0[0])
                elif method == 'UCC':
                    Grad = self.gradient_ucc(parameters,ioff,ioff+nparams)
                elif method == 'ADAPT':
                    Grad = self.gradient_adapt(self._ops[i],params[i],bra[i],ket[i])
                    if not self._normalized_wf:
                        Ovlp = self.gradient_adapt(self._ops[i],params[i],v0[i],ket[i])
                elif method == 'TAYLOR':
                    Grad = self.gradient_taylor(self._ops[i],params[i],bra[i],ket[i-1])
                    if not self._normalized_wf:
                        Ovlp = self.gradient_taylor(self._ops[i],params[i],v0[i],ket[i-1])
                else:
                    print('Invalid Transformation Type')
                    exit()
                ioff += nparams

                #print(Grad,Ovlp)
                for i in range(len(Grad)):
                    if not self._normalized_wf:
                        Grad[i] = (Grad[i]-E*Ovlp[i])/S
                    Grad0.append(Grad[i])

        #print(parameters,E,S)
        #print(Grad0)
        #Grad0 = self.gradient_finite_difference(parameters)
        #print(Grad0)

        Grad = np.asarray(Grad0)
        self._max_grad = np.max(np.abs(Grad))

        return np.asarray(Grad)


    def gradient_finite_difference(self, params):
        '''
        '''
        grad = []
        _eps = 1e-5
        for i in range(len(params)):
            params1 = copy.deepcopy(params)
            params1[i] = params1[i] + _eps
            e1 = self.energy(params1)
            params1[i] = params1[i] - _eps*2
            e2 = self.energy(params1)
            grad.append((e1-e2)/(2*_eps))

        Grad = np.asarray(grad)
        self._max_grad = np.max(np.abs(Grad))

        return Grad


    def update_operators(self, params, nadd):
        '''
        '''
        grad = self.residual_gradient(params,nadd)

        theta_thresh = self._gtol_res
        gg = np.abs(grad)
        idx = np.nonzero(gg>theta_thresh)[0]
        idx2 = np.argsort(gg[idx])
        idx2 = idx2[::-1]
        #print(gg[idx[idx2[:10]]])
        #print(idx[idx2[:10]])
        if type(self.pool) is FermionOps and (self.pool._isym == 'sc' or self.pool._isym == 'sa'):
            list1 = list(idx[idx2[:min(nadd,len(idx))]])
        else:
            list1 = []
            for i in range(len(idx)):
                to_add = True
                index1 = self.pool.get_index(idx[idx2[i]])
                #print(self.pool._ops[idx[idx2[i]]])
                for j in list1:
                    if self.pool.get_index(j)==index1:
                        to_add = False; break
                if to_add: list1.append(idx[idx2[i]])
                if len(list1) == min(nadd,len(idx)): break

        return list1

            
    def update(self, ops=None, params=None, ops_mat=None):
        '''
        '''
        print('Should overwite update function')
        exit()


    def callback(self,x):
        '''
        '''
        if self._print_level > 4:
            print(" %4i  %20.10f        %10.1e" %(self._niter, self._energy, self._max_grad))
        elif self._print_level > 1:
            if self._niter < 4:
                print(" %4i  %20.10f        %10.1e" %(self._niter, self._energy, self._max_grad))
            elif (self._niter-4)%35 == 0:
                print('      .',end='')
            elif (self._niter-4)%35 == 34:
                print('.')
            else:
                print('.',end='')
        self._niter += 1
        return sys.stdout.flush()


    def get_number_of_hamiltonian_terms(self, parameters):
        '''
        '''
        params = self.parse_params(parameters)
        qubit_ham = copy.deepcopy(self.scf._qubit_ham)
        for i in reversed(range(len(params))):
            if self._params_group[i][2].upper() == 'TAYLOR':
                U0 = QubitOperator.identity()
                for j in range(len(params[i])):
                    opA = self.pool._ops[self._ops[i][j]]
                    if isinstance(opA,FermionOperator):
                        qubit_ops = jordan_wigner(opA)
                    else:
                        qubit_ops = opA
                    U0 = U0 + params[i][j]*qubit_ops
                qubit_ham = hermitian_conjugated(U0)*qubit_ham*U0
                qubit_ham.compress(1e-15)
                for term in qubit_ham.terms:
                    print(term, qubit_ham.terms[term])


        return len(qubit_ham.terms)
