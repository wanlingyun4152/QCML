import scipy
import numpy as np
import copy
from openfermion import jw_configuration_state


class HFRefState():
    ''' Hartree-Fock reference state
    '''
    def __init__(self, scf, name='HF'):
        '''
        '''
        self._name = name
        self._energy = None
        self._state = None
        self._params = []
        self._params_group = [[0,True,'REF']]

        self.scf = scf
        self.kernel()

    def energy(self):
        ''' E = <v|H|v>
        '''
        v = self.state()
        E = v.conj().T.dot(self.scf._ham.dot(v))[0,0]
        self._energy = E.real
        return E.real

    def state(self):
        ''' |HF> = |1100>
        '''
        if self._state is None:
            occ = []
            for i in range(self.scf._noa): occ.append(i*2)
            for i in range(self.scf._nob): occ.append(i*2+1)
            HF_state = jw_configuration_state(occ, self.scf._nqubit)
            HF_state = scipy.sparse.csc_matrix(HF_state).transpose()
            self._state = HF_state
        else:
            HF_state = self._state

        return HF_state

    def kernel(self):
        '''
        '''
        self._energy = self.energy()
        print(' %s energy: %16.10f' %(self._name,self._energy))

    def gradient(self, params=None, bra=None):
        '''
        '''
        return []


class HFTaperState(HFRefState):
    '''
    '''
    def state(self):
        ''' |HF> = |1001>
        '''
        if self._state is None:
            nqubit = self.scf._norb*2
            noa = self.scf._noa
            nob = self.scf._nob
            hf_array = np.zeros(nqubit)
            hf_array[::2] = np.hstack([np.ones(noa), np.zeros(nqubit//2-noa)])
            hf_array[1::2] = np.hstack([np.ones(nob), np.zeros(nqubit//2-nob)])
            ref_state=hf_array.astype(int)
            ref_taper = ref_state[self.scf._free_qubit_indices]

            from symmer import QuantumState
            print(f'Tapered off HF state: {ref_taper}')
            ref_psi_tap = QuantumState([ref_taper])
            self._state = ref_psi_tap.to_sparse_matrix

        return self._state


class CISRefState(HFRefState):
    '''
    '''
    def state(self):
        ''' |HF> = |1001>
        '''
        if self._state is None:
            occ = []
            for i in range(self.scf._noa): occ.append(i*2)
            for i in range(self.scf._nob): occ.append(i*2+1)
            occ[self.scf._noa+self.scf._nob-1] = self.scf._noa+self.scf._nob
            HF_state = jw_configuration_state(occ, self.scf._nqubit)
            HF_state = scipy.sparse.csc_matrix(HF_state).transpose()
            self._state = HF_state
        else:
            HF_state = self._state

        return HF_state


class RefStateFromInput(HFRefState):
    '''
    '''
    def __init__(self, scf, name='input', v=None):
        ''' Set the reference state to an input state ''v''
        '''
        self._name = name
        self._energy = None
        self._state = v
        self._params = []
        self._params_group = [[0,True,'REF']]

        self.scf = scf
        self.kernel()


    def state(self):
        '''
        '''
        if self._state is None:
            print(' Please initionalize the reference state!')
            exit()
        return self._state


class QMFRefState():
    ''' Qubit mean-field reference state
        |v> = cos(theta/2)|0> + e^(i phi) sin(theta/2) |1>
        |0> = (1 0), |1> = (0 1) 
    '''
    def __init__(self, scf, name='QMF', fix=True):
        '''
        '''
        self._name = name
        self.scf = scf
        self._energy = None
        self._state = None
        self._params = self.kernel()
        self._params_group = [[2*self.scf._nqubit,fix,'REF']]

    def energy(self, params=None):
        ''' E = <v|H|v>
        '''
        v = self.state(params)
        E = v.conj().T.dot(self.scf._ham.dot(v))[0,0]
        self._energy = E.real
        return E.real

    def state(self, params):
        ''' Overwrite Hartree-Fock state
        '''
        QMF_state = 1
        for i in range(self.scf._nqubit):
            theta = params[i*2]
            phi = params[i*2+1]
            v = np.array((np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)))
            QMF_state = np.kron(QMF_state,v)
        QMF_state = scipy.sparse.csc_matrix(QMF_state).transpose()
        self._state = QMF_state
        return QMF_state

    def kernel(self):
        ''' Optimize variational parameters
        '''
        print('******************************************************')
        print('                  Quantum Mean Field                  ')
        print('                                                      ')
        params = [0.0]*(self.scf._nqubit*2)
        for i in range(self.scf._nqubit):
            params[2*i] = 1.0
        result = scipy.optimize.minimize(self.energy, params, method='BFGS', jac=self.gradient)
        e = self.energy(result['x'])
        v = self.state(result['x'])
        state = []
        for i in range(self.scf._nqubit):
            theta = result['x'][i*2]
            phi   = result['x'][i*2+1]
            state.append([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
        print('          QMF energy: %16.10f' %e )
        print('          <S^2>     : %16.10f'  %(v.conj().T.dot(self.scf._s2.dot(v))[0,0].real))
        print('                       STATE                          ')         
        print(state)
        print('******************************************************')
        print('\n')
  
        self._energy = e
        self._state = v       
        return result['x']

    def gradient(self, params, bra=None):
        ''' Gradient of QMF or QMF as a reference state

        Args: 
            bra: <v|H or <v|U'HU 

        Output:
            grad: \partial E / \partial (theta or phi) 
        '''
        if bra is None: bra = (self.scf._ham.dot(self.state(params))).conj().T

        grad = []
        QMF_state = 1
        for i in range(self.scf._nqubit):
            theta = params[i*2]
            phi   = params[i*2+1]
            v0 = np.array((np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)))
            v1 = np.array((-np.sin(theta/2)/2, np.exp(1j*phi)*np.cos(theta/2)/2))
            v1 = np.kron(QMF_state,v1)
            v2 = np.array((0,1j*np.exp(1j*phi)*np.sin(theta/2)))
            v2 = np.kron(QMF_state,v2)
            QMF_state = np.kron(QMF_state,v0)
            for j in range(i+1,self.scf._nqubit):
                theta = params[j*2]
                phi   = params[j*2+1]
                v0 = np.array((np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)))
                v1 = np.kron(v1,v0)
                v2 = np.kron(v2,v0)
            grad.append((bra.dot(v1)[0].real)*2)
            grad.append((bra.dot(v2)[0].real)*2)
        return np.asarray(grad)

    def gradient_finite_difference(self, params):
        ''' Gradient of QMF or QMF as a reference state 
            using the finite difference method

        Output:
            grad: \partial E / \partial (theta or phi) 
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

        return grad
