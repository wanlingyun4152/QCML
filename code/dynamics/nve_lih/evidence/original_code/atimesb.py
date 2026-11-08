import numpy as np
import scipy
from quspin.tools.expm_multiply_parallel_core import expm_multiply_parallel as exp
from quspin.tools.expm_multiply_parallel_core import csr_matvec

def expm_multiply(A,B):

    dtype_real = np.float64
    dtype_cmplx = np.result_type(dtype_real, np.complex64)

    psi=B.toarray().flatten().copy().astype(np.complex128)
    work_array=np.zeros((2*psi.shape[0],), dtype=psi.dtype) # twice as long because complex-valued

    eA = exp(A, a=1.0, dtype=dtype_cmplx)

    eA.dot(psi,work_array=work_array,overwrite_v=True)

    return scipy.sparse.csc_matrix(psi.reshape((psi.shape[0],1)))

def expect_value(bra,ket,op):
    
    psi = csr_matvec(op,ket.toarray())
    value = bra.transpose().conj().dot(scipy.sparse.csc_matrix(psi.reshape((psi.shape[0],1))))[0,0]

    return value

def csr_matvec_a(op,ket):

    psi = csr_matvec(op,ket.toarray())
    return scipy.sparse.csc_matrix(psi.reshape((psi.shape[0],1)))


