from ansatz import Ansatz
from tools import read_ansatz
from rdm import ACSE
import copy as cp
import numpy as np
import random
import os


def ADAPT(options, hf, pool, ref):
    ''' Adaptive Wave Function Ansatz
    '''
    itype = options['method'].lower()
    form = options['form'].lower()

    if itype=='adapt': 
        if form=='unitary':
            ansatz = ADAPTu(options, hf, pool, ref)
        else: 
            ansatz = ADAPTt(options, hf, pool, ref)
    elif itype=='tc1':
        ansatz = TC1(options, hf, pool, ref)
    elif itype=='tc2':
        ansatz = TC2(options, hf, pool, ref)
    else:
        print(' Check Wave Function Ansatz %s' %itype)
        print(' Available: ucc, adapt, tc1, tc2')
        exit()

    return ansatz


class UCC(Ansatz):
    ''' Unitary Coupled Cluster Ansatz
    '''
    def update(self):
        ''' Update Operators and Parameters
        '''
        nops = len(self.pool._ops)
        self._ops.append([i for i in range(nops)])
        self._params.append([0]*nops)
        self._params_group.append([nops,False,'UCC'])
        self._max_res_grad = 0
        self._norm_res_grad = 0


class Custom(Ansatz):
    ''' Customized Wave Function Ansatz
    '''
    def update(self):
        '''
        '''
        ansatz = read_ansatz()

        if ansatz['class'].upper() != self.pool._class:
            print('Input operator pool class: %s, readin class: %s' 
                       %(ansatz['class'].upper(),self.pool._class))
            raise

        self._ops = ansatz['ops_list']
        self._params_group = ansatz['group']
        self._params = []
        for group in self._params_group:
            self._params.append([0.1]*group[0])
            if group[2].lower()=='taylor':
                self._normalized_wf = False
        self._max_res_grad = 0
        self._norm_res_grad = 0

        if self._print_level > 0:
            if self._print_level > 4:
                print(self._params_group)
                print(self._ops)
                print(self._params)


class ADAPTu(Ansatz):
    ''' Adaptive Wave Function Ansatz (Unitary)
    '''
    _acse_grad = False
    _acse = ACSE()

    def residual_gradient(self, parameters, nadd=None):
        ''' Residual Gradient for Updating Wave Function Ansatz
        '''
        nops = len(self.pool._ops)
        # v: quantum state; Hv: Hamiltonian acting on state
        v = self.state(parameters)
        Hv = self.scf._ham.dot(v)
        if self._acse_grad:
            # Residual gradient from ACSE approximation
            grad = self._acse.get_acse(v)
        else:
            # Residual gradient <v|[tau, H]|v>
            grad = np.zeros(nops)
            for i in range(nops):
                grad[i] = self.pool.gradient(i,v,Hv)
        
        self._max_res_grad = np.max(np.abs(grad))
        self._norm_res_grad = np.linalg.norm(grad)
        if self._print_level > 0:
            print('Maximum Absolute Residue Gradient: %10.6f' %self._max_res_grad)

        return grad


    def update(self, params, niter=0):
        ''' Updating Wave Function Ansatz
        '''
        # Choosing 'Nu' operators to update wave function ansatz
        list1 = self.update_operators(params, self._options['Nu'])
 
        self._niter = 0

        if niter == 0:
            self._ops.append(list1)
            self._params_group.append([len(list1),False,'adapt'])
            for i in range(len(list1)):
                params = np.append(params,0)
            self._params.append([0]*len(list1))
        else:
            self._params_group[1][0] += len(list1)
            for i in range(len(list1)):
                self._ops[1].append(list1[i])
                params = np.append(params,0)
                self._params[1] = np.append(self._params[1],0)

        if self._options['random']:
            for i in range(len(params)):
                params[i] = random.random()/10

        # Print operator strings for debug
        if self._print_level > 0:
            if self._print_level > 4:
                print(self._params_group)
                print(self._ops)
                print(self._params)
            for i in range(len(list1)):
                print(self.pool._ops[list1[i]])

        return params


class ADAPTt(Ansatz):
    ''' Adaptive Wave Function Ansatz (Taylor)
    '''
    def residual_gradient(self, parameters, nadd=None):
        ''' Residual Gradient for Growing Unnormalized Wave Function
        '''
        nops = len(self.pool._ops)
        # v: quantum state; Hv: Hamiltonian acting on state
        v = self.state(parameters)
        Hv = self.scf._ham.dot(v)

        # Residual gradient (<v|[tau, H]|v> <v|v> - <v|tau|v>*E) / <v|v>**2
        grad = np.zeros(nops)
        for i in range(nops):
            grad[i] = self.pool.gradient(i,v,Hv)
        
        self._max_res_grad = np.max(np.abs(grad))
        self._norm_res_grad = np.linalg.norm(grad)
        if self._print_level > 0:
            print('Maximum Absolute Residue Gradient: %10.6f' %self._max_res_grad)

        return grad


    def update(self, params, niter=0):
        '''
        '''
        list1 = self.update_operators(params, self._options['Nt'])
 
        self._niter = 0
        self._normalized_wf = False

        self._ops.append(list1)
        self._params_group.append([len(list1),False,'taylor'])
        for i in range(len(list1)):
            params = np.append(params,0)
        self._params.append([0]*len(list1))

        if self._print_level > 0:
            if self._print_level > 4:
                print(self._params_group)
                print(self._ops)
                print(self._params)
            for i in range(len(list1)):
                print(self.pool._ops[list1[i]])

        return params

        
class TC1(Ansatz):    
    ''' Transcorrelated Method
    '''  
    def residual_gradient(self, parameters, nadd=None):
        ''' Residual Gradient for Growing Wave Function
        '''
        nops = len(self.pool._ops)
        if not self._normalized_wf:
            Hv, v = self.get_Hv(parameters,nadd+1)
            ovlp, v = self.get_Hv(parameters,nadd+1,0,True)
            E = self.energy(parameters)
            S = ovlp.conj().T.dot(v)[0,0].real
        else:
            v = self.state(parameters)
            Hv = self.scf._ham.dot(v)

        grad = np.zeros(nops)
        for i in range(nops):
            opA = self.pool.ops_mat(i)
            gi = 2*(Hv.conj().T.dot(opA.dot(v)))[0,0]
            if not self._normalized_wf:
                si = 2*(ovlp.conj().T.dot(opA.dot(v)))[0,0]
                gi = (gi-E*si)/S
            grad[i] = gi.real
        
        self._max_res_grad = np.max(np.abs(grad))
        self._norm_res_grad = np.linalg.norm(grad)
        if self._print_level > 0:
            print('Maximum Absolute Residue Gradient: %10.6f' %self._max_res_grad)

        return grad


    def update(self, params, niter):
        '''
        '''
        Nu = self._options['Nu'] 
        self._niter = 0
        if niter < Nu:
            list1 = self.update_operators(params, 1)
            self._ops.append(list1)
            self._params_group.append([len(list1),False,'adapt'])
            for i in range(len(list1)):
                params = np.append(params,0)
            self._params.append([0]*len(list1))
        else:
            list1 = self.update_operators(params, self._options['Nt'])
            self._normalized_wf = False
            self._ops.insert(Nu+1,list1)
            self._params_group.insert(Nu+1,[len(list1),False,'taylor'])            
            self._params.insert(Nu+1,[0]*len(list1))

            if not self._options['OptAll']:
                if niter > Nu: self._params_group[Nu+2][1] = True
            
            params = []
            for i in range(len(self._params)):
                if not self._params_group[i][1]:
                    for j in range(self._params_group[i][0]):
                        params.append(self._params[i][j])
            params = np.asarray(params)


        if self._print_level > 0:
            if self._print_level > 5:
                print(self._params_group)
                print(self._ops)
                print(self._params)
            for i in range(len(list1)):
                print(self.pool._ops[list1[i]])

        return params


class TC2(Ansatz):    
    ''' Transcorrelated Method
    '''  
    def residual_gradient(self, parameters, nadd=None):
        ''' Residual Gradient for Growing Wave Function
        '''
        nops = len(self.pool._ops)
        v = self.state(parameters)
        Hv = self.scf._ham.dot(v)

        grad = np.zeros(nops)
        for i in range(nops):
            grad[i] = self.pool.gradient(i,v,Hv)
        
        self._max_res_grad = np.max(np.abs(grad))
        self._norm_res_grad = np.linalg.norm(grad)
        if self._print_level > 0:
            print('Maximum Absolute Residue Gradient: %10.6f' %self._max_res_grad)

        return grad


    def update(self, params, niter):
        '''
        '''
        Nu = self._options['Nu'] 
        if niter < Nu:
            # add one operator (Unitary) for updating wave function Ansatz
            list1 = self.update_operators(params, 1)
            itype = 'adapt'
        else:
            # add 'Nt' operators (Taylor expansion) for dressing Hamiltonian
            list1 = self.update_operators(params, self._options['Nt'])
            itype = 'taylor'
            self._normalized_wf = False
       
        self._niter = 0
        self._ops.append(list1)
        self._params_group.append([len(list1),False,itype])
        self._params.append([0]*len(list1))

        #if 'Nopt' in self._options:
        #    nopt = self._options['Nopt']
        #else:
        #    nopt = 1
        #nopt -= 1
        #if not self._options['OptAll'] and niter >= Nu+nopt: 
        #    if niter > Nu: self._params_group[niter-nopt][1] = True

        #    m = 0
        #    for i in range(self._options['Nu']+1):
        #        m += self._params_group[i][0]
        #    params = params[0:m]

        # unfold 2D array into 1D array
        params = []
        for i in range(len(self._params)):
            if not self._params_group[i][1]:
                for j in range(self._params_group[i][0]):
                    params.append(self._params[i][j])
        params = np.asarray(params)

        if self._print_level > 0:
            if self._print_level > 5:
                print(self._params_group)
                print(self._ops)
                print(self._params)
            for i in range(len(list1)):
                print(self.pool._ops[list1[i]])

        return params


