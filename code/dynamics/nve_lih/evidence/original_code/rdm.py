import numpy as np
from scipy.special import comb
from itertools import combinations_with_replacement as Cmn
from atimesb import expect_value
from openfermion.linalg.wedge_product import wedge
from openfermion import FermionOperator
from openfermion.linalg import get_sparse_operator

# op = a_i^+ a_j
def one_rdm(psi,norb,nqubit):

    RDM = np.zeros((nqubit,nqubit))
    #RDM = np.zeros((nqubit,nqubit),dtype=np.complex_)
    for i in range(nqubit):
        for j in range(nqubit):
            op = get_1e_op(i,j,nqubit)
            #RDM[i,j] = psi.T.conj().dot(op.dot(psi))[0,0].real
            RDM[i,j] = psi.T.conj().dot(op.dot(psi))[0,0].real
    return RDM

# op = a_i^+ a_j^+ a_l a_k
# D^ij_kl = <\psi| op | \psi>/2
def two_rdm(psi,norb,nqubit):
   
    RDM = np.zeros((nqubit,nqubit,nqubit,nqubit))
    
    pqrs = 0
    ij = -1
    for i in range(nqubit):
        for j in range(i+1,nqubit):
            ij += 1
            kl = -1
            for k in range(nqubit):
                for l in range(k+1,nqubit):
                    kl += 1
                    if((ij<=kl) and ((i%2+j%2)==(k%2+l%2))):
                        pqrs += 1
                        op = get_2e_op(i,j,k,l,nqubit)
                        #aa = (psi.T.conj().dot(op.dot(psi))[0,0].real)/2.0
                        aa = expect_value(psi,psi,op).real/2.0
                        # op = a_i^+ a_j^+ a_l a_k
                        RDM[i,j,k,l] = aa 
                        # op = a_i^+ a_j^+ a_k a_l
                        RDM[i,j,l,k] = -aa 
                        # op = a_j^+ a_i^+ a_k a_l 
                        RDM[j,i,l,k] = aa
                        # op = a_j^+ a_i^+ a_l a_k
                        RDM[j,i,k,l] = -aa 
                        # 
                        aa_conj = np.conj(aa) 
                        RDM[k,l,i,j] = aa_conj
                        #
                        RDM[l,k,i,j] = -aa_conj
                        #
                        RDM[l,k,j,i] = aa_conj
                        #
                        RDM[k,l,j,i] = -aa_conj

    print("number of 2-RDM: %10d" %pqrs)
    return RDM


def cumulant_two_rdm(RDM1,RDM2):

    delta = RDM2 - np.einsum("ik,jl->ijkl",RDM1,RDM1)/2
    delta+= np.einsum("il,jk->ijkl",RDM1,RDM1)/2

    return delta


def cumulant_three_rdm(RDM1,RDM2,occ):
 
    D2 = cumulant_two_rdm(RDM1,RDM2)

    delta = np.einsum("l,ilqs,jklt->ijkqst",occ,D2,D2)/6
    delta-= np.einsum("l,jlqs,iklt->ijkqst",occ,D2,D2)/6
    delta-= np.einsum("l,klqs,jilt->ijkqst",occ,D2,D2)/6
    delta-= np.einsum("l,ilts,jklq->ijkqst",occ,D2,D2)/6
    delta-= np.einsum("l,ilqt,jkls->ijkqst",occ,D2,D2)/6
    delta+= np.einsum("l,jlts,iklq->ijkqst",occ,D2,D2)/6
    delta+= np.einsum("l,jlqt,ikls->ijkqst",occ,D2,D2)/6
    delta+= np.einsum("l,klts,jilq->ijkqst",occ,D2,D2)/6
    delta+= np.einsum("l,klqt,jils->ijkqst",occ,D2,D2)/6

    return delta

# op = a_i^+ a_j^+ a_k^+ a_l a_m a_n
def three_rdm(psi,norb,nqubit):
   
    RDM = np.zeros((nqubit,nqubit,nqubit,nqubit,nqubit,nqubit))
    for i in range(nqubit):
        for j in range(i+1,nqubit):
            for k in range(nqubit):
                for l in range(nqubit):
                    for m in range(l+1,nqubit):
                        for n in range(nqubit):
                            op = get_3e_op(i,j,k,l,m,n,nqubit)
                            #aa = (psi.T.conj().dot(op.dot(psi))[0,0].real)/6.0
                            aa = expect_value(psi,psi,op).real/6.0
                            RDM[i,j,k,l,m,n] = aa 
                            RDM[j,i,k,l,m,n] = -aa 
                            RDM[j,i,k,m,l,n] = aa  
                            RDM[i,j,k,m,l,n] = -aa 

    return RDM

# op = a_i^+ a_j^+ a_k^+ a_l a_m a_n
def three_rdm2(psi,norb,nqubit):
   
    RDM = np.zeros((nqubit,nqubit,nqubit,nqubit,nqubit,nqubit))
    for ijk in Cmn(range(nqubit),3):
        for lmn in Cmn(range(nqubit),3):
            i = int(ijk[0])
            j = int(ijk[1])
            k = int(ijk[2])
            l = int(lmn[0])
            m = int(lmn[1])
            n = int(lmn[2])
            op = get_3e_op(i,j,k,l,m,n,nqubit)
            aa = expect_value(psi,psi,op).real/6.0
            #aa = (psi.T.conj().dot(op.dot(psi))[0,0].real)/6.0
            RDM[i,j,k,l,m,n] = aa
            RDM[i,k,j,l,m,n] = -aa
            RDM[j,i,k,l,m,n] = -aa
            RDM[k,i,j,l,m,n] = aa
            RDM[j,k,i,l,m,n] = aa
            RDM[k,j,i,l,m,n] = -aa

            bb = -aa
            RDM[i,j,k,m,l,n] = bb
            RDM[i,k,j,m,l,n] = -bb
            RDM[j,i,k,m,l,n] = -bb
            RDM[k,i,j,m,l,n] = bb
            RDM[j,k,i,m,l,n] = bb
            RDM[k,j,i,m,l,n] = -bb

            bb = aa
            RDM[i,j,k,m,n,l] = bb
            RDM[i,k,j,m,n,l] = -bb
            RDM[j,i,k,m,n,l] = -bb
            RDM[k,i,j,m,n,l] = bb
            RDM[j,k,i,m,n,l] = bb
            RDM[k,j,i,m,n,l] = -bb

            bb = -aa
            RDM[i,j,k,n,m,l] = bb
            RDM[i,k,j,n,m,l] = -bb
            RDM[j,i,k,n,m,l] = -bb
            RDM[k,i,j,n,m,l] = bb
            RDM[j,k,i,n,m,l] = bb
            RDM[k,j,i,n,m,l] = -bb

            bb = -aa
            RDM[i,j,k,l,n,m] = bb
            RDM[i,k,j,l,n,m] = -bb
            RDM[j,i,k,l,n,m] = -bb
            RDM[k,i,j,l,n,m] = bb
            RDM[j,k,i,l,n,m] = bb
            RDM[k,j,i,l,n,m] = -bb

            bb = aa
            RDM[i,j,k,n,l,m] = bb
            RDM[i,k,j,n,l,m] = -bb
            RDM[j,i,k,n,l,m] = -bb
            RDM[k,i,j,n,l,m] = bb
            RDM[j,k,i,n,l,m] = bb
            RDM[k,j,i,n,l,m] = -bb

    return RDM


def three_rdm_V(RDM1,RDM2,norb,nqubit):

    D2 = cumulant_two_rdm(RDM1,RDM2)
    print("trace of RDM1: %10.8f" %(one_rdm_trace(RDM1,nqubit)))
    print("trace of RDM2: %10.8f" %(two_rdm_trace(RDM2,nqubit)))
    print("trace of delta RDM2: %10.8f" %(two_rdm_trace(D2,nqubit)))
    
    RDM = np.zeros((nqubit,nqubit,nqubit,nqubit,nqubit,nqubit))
    for i in range(nqubit):
        for j in range(i,nqubit):
        #for j in range(nqubit):
            for k in range(nqubit):
                for q in range(nqubit):
                    for s in range(q,nqubit):
                    #for s in range(nqubit):
                        for t in range(nqubit):
                            a  =  0 
                            a +=  RDM1[i,q]*RDM1[j,s]*RDM1[k,t]/2    
                            a +=  RDM1[i,t]*RDM1[j,q]*RDM1[k,s]/2    
                            a +=  RDM1[i,s]*RDM1[j,t]*RDM1[k,q]/2    
                            a -=  RDM1[i,s]*RDM1[j,q]*RDM1[k,t]/2    
                            a -=  RDM1[i,t]*RDM1[j,s]*RDM1[k,q]/2    
                            a -=  RDM1[i,q]*RDM1[j,t]*RDM1[k,s]/2    
                            a +=  RDM1[i,q]*D2[j,k,s,t] 
                            a +=  RDM1[j,s]*D2[i,k,q,t] 
                            a +=  RDM1[k,t]*D2[i,j,q,s] 
                            a -=  RDM1[j,q]*D2[i,k,s,t] 
                            a -=  RDM1[i,s]*D2[j,k,q,t] 
                            a -=  RDM1[k,q]*D2[i,j,t,s] 
                            a -=  RDM1[i,t]*D2[j,k,s,q] 
                            a -=  RDM1[k,s]*D2[i,j,q,t] 
                            a -=  RDM1[j,t]*D2[i,k,q,s] 
                            a *= 1/3
                            RDM[i,j,k,q,s,t] = a
                            RDM[j,i,k,q,s,t] = -a
                            RDM[i,j,k,s,q,t] = -a
                            RDM[j,i,k,s,q,t] = a
    return RDM

def three_rdm_V2(RDM1,RDM2,norb,nqubit):
    
    a = np.zeros((nqubit,nqubit,nqubit,nqubit,nqubit,nqubit))
    a -= np.einsum('iq,js,kt->ijkqst',RDM1,RDM1,RDM1)
    a -= np.einsum('it,jq,ks->ijkqst',RDM1,RDM1,RDM1) 
    a -= np.einsum('is,jt,kq->ijkqst',RDM1,RDM1,RDM1) 
    a += np.einsum('is,jq,kt->ijkqst',RDM1,RDM1,RDM1) 
    a += np.einsum('it,js,kq->ijkqst',RDM1,RDM1,RDM1) 
    a += np.einsum('iq,jt,ks->ijkqst',RDM1,RDM1,RDM1) 
    a += np.einsum('iq,jkst->ijkqst',RDM1,RDM2)
    a += np.einsum('js,ikqt->ijkqst',RDM1,RDM2)
    a += np.einsum('kt,ijqs->ijkqst',RDM1,RDM2)
    a -= np.einsum('jq,ikst->ijkqst',RDM1,RDM2)
    a -= np.einsum('is,jkqt->ijkqst',RDM1,RDM2)
    a -= np.einsum('kq,ijts->ijkqst',RDM1,RDM2)
    a -= np.einsum('it,jksq->ijkqst',RDM1,RDM2)
    a -= np.einsum('ks,ijqt->ijkqst',RDM1,RDM2)
    a -= np.einsum('jt,ikqs->ijkqst',RDM1,RDM2)
    a = a/3.0
    return a

def three_rdm_V3(RDM1,RDM2,norb,nqubit):
    
    D2 = cumulant_two_rdm(RDM1,RDM2)
    RDM = np.zeros((nqubit,nqubit,nqubit,nqubit,nqubit,nqubit))
    RDM = wedge(RDM1,wedge(RDM1,RDM1,(1,1),(1,1)),(1,1),(2,2))
    RDM += 3*wedge(D2,RDM1,(2,2),(1,1))

    return RDM
 

def one_rdm_trace(RDM1,nqubit):

    a = 0 
    for i in range(nqubit):
        a += RDM1[i,i]
    return a
 


def two_rdm_trace(RDM2,nqubit):

    a = 0 
    for i in range(nqubit):
        for j in range(nqubit):
            a += RDM2[i,j,i,j]
    return a
 


def three_rdm_trace(RDM3,nqubit):

    a = 0 
    for i in range(nqubit):
        for j in range(nqubit):
            for k in range(nqubit):
                a += RDM3[i,j,k,i,j,k]
    return a
               
                            
def get_1e_op(i,j,nqubit):

    op = openfermion.FermionOperator(((i,1),(j,0)),1.0)

    return openfermion.linalg.get_sparse_operator(op,nqubit)


def get_2e_op(i,j,k,l,nqubit):

    op = openfermion.FermionOperator(((i,1),(j,1),(l,0),(k,0)),1.0)

    return openfermion.linalg.get_sparse_operator(op,nqubit)

   
def get_3e_op(i,j,k,l,m,n,nqubit):

    op = openfermion.FermionOperator(((i,1),(j,1),(k,1),(n,0),(m,0),(l,0)),1.0)

    return openfermion.linalg.get_sparse_operator(op,nqubit)

def get_spin_ham(h1,h2,norb,nqubit):
    
    h11 = np.zeros((nqubit,nqubit))
    for p in range(norb):
        pa = 2*p
        pb = 2*p+1
        for q  in range(norb):
            qa = 2*q
            qb = 2*q+1
            h11[pa,qa] = h1[p,q]
            h11[pb,qb] = h1[p,q]

    h22 = np.zeros((nqubit,nqubit,nqubit,nqubit))
    for p in range(norb):
        pa = 2*p 
        pb = 2*p+1 
        for q in range(norb):
            qa = 2*q 
            qb = 2*q+1 
            for r in range(norb):
                ra = 2*r 
                rb = 2*r+1 
                for s in range(norb):
                    sa = 2*s 
                    sb = 2*s+1 
                    h22[pa,qa,ra,sa] = h2[p,r,q,s]
                    h22[pa,qb,ra,sb] = h2[p,r,q,s]    
                    h22[pb,qa,rb,sa] = h2[p,r,q,s]
                    h22[pb,qb,rb,sb] = h2[p,r,q,s]
    return h11,h22

def get_acse(psi,ham,pool,ne,norb,nqubit,method=0,spin_adapted=1):
    
    h11 = ham.int_H
    h22 = ham.int_V
    h1,h2 = get_spin_ham(h11,h22,norb,nqubit)
   
    RDM1 = one_rdm(psi,norb,nqubit)
    RDM2 = two_rdm(psi,norb,nqubit)
    #E = np.einsum('ij,ij',h1,RDM1)
    #E+= np.einsum('ijkl,ijkl',h2,RDM2)
    fermi_ham  = ham.export_FermionOperator()
    hamiltonian = openfermion.linalg.get_sparse_operator(fermi_ham)
    #E2 = psi.T.conj().dot(hamiltonian.dot(psi))[0,0].real
    #print("energy:%10.6f %10.6f" %(E,E2))
    print("two RDM trace %10.4f" %(two_rdm_trace(RDM2,nqubit)))
    if method==0:
        RDM3 = three_rdm(psi,norb,nqubit)
    elif method==1 or method==4:
        print("approximate 3rd RDM")
        #RDM1 = one_rdm(psi,norb,nqubit)
        RDM3 = three_rdm_V(RDM1,RDM2,norb,nqubit)
        if method==4: 
            occ = np.ones((nqubit))
            occ[ne:nqubit] = -1
            RDM3 += cumulant_three_rdm(RDM1,RDM2,occ) 
    elif method==2:
        RDM3 = three_rdm2(psi,norb,nqubit)
    trace_rdm3 = three_rdm_trace(RDM3,nqubit)
    RDM3 = RDM3*(comb(ne,3)/trace_rdm3)
    print("three RDM trace %10.4f, after corrected %10.4f" %(trace_rdm3,three_rdm_trace(RDM3,nqubit)))

 
    # aa= <\psi | [a_i^+ a_j, H] |\psi>
    aa  = np.einsum('jk,ik->ij',h1,RDM1)
    aa -= np.einsum('ki,kj->ij',h1,RDM1)
    aa += np.einsum('jklm,iklm->ij',h2,RDM2)*2.0
    aa += np.einsum('klmi,kljm->ij',h2,RDM2)*2.0

    # a = <\psi | [a_i^+ a_j^+ a_l a_k, H] |\psi>
    a  = np.einsum('ks,ijsl->ijlk',h1,RDM2)*2.0
    a -= np.einsum('ls,ijsk->ijlk',h1,RDM2)*2.0
    a += np.einsum('pj,pikl->ijlk',h1,RDM2)*2.0
    a -= np.einsum('pi,pjkl->ijlk',h1,RDM2)*2.0
    a += np.einsum('pkst,ijpstl->ijlk',h2,RDM3)*6.0
    a -= np.einsum('plst,ijpstk->ijlk',h2,RDM3)*6.0
    a += np.einsum('pqsj,pqikls->ijlk',h2,RDM3)*6.0
    a -= np.einsum('pqsi,pqjkls->ijlk',h2,RDM3)*6.0
    a += np.einsum('klst,ijst->ijlk',h2,RDM2)*2.0
    a += np.einsum('pqji,pqkl->ijlk',h2,RDM2)*2.0 

    #fermi_ham  = ham.export_FermionOperator()
    #hamiltonian = openfermion.linalg.get_sparse_operator(fermi_ham)
    #grad2 = np.zeros((nqubit,nqubit))
    #for i in range(nqubit):
    #    for j in range(nqubit):
    #        op = get_1e_op(i,j,nqubit)
    #        bb = (psi.T.conj().dot(op.dot(hamiltonian.dot(psi)))[0,0].real)
    #        bb -= (psi.T.conj().dot(hamiltonian.dot(op.dot(psi)))[0,0].real)
    #        grad2[i,j] = bb
    #dgrad = aa - grad2
    #print("\n grad difference")
    #print(dgrad[np.abs(dgrad)>1e-8])


    #fermi_ham  = ham.export_FermionOperator()
    #hamiltonian = openfermion.linalg.get_sparse_operator(fermi_ham)
    #grad2 = np.zeros((nqubit,nqubit,nqubit,nqubit))
    #for i in range(nqubit):
    #    for j in range(i,nqubit):
    #        for k in range(nqubit):
    #            for l in range(k,nqubit):
    #                op = get_2e_op(i,j,k,l,nqubit)
    #                aa = (psi.T.conj().dot(op.dot(hamiltonian.dot(psi)))[0,0].real)
    #                aa -= (psi.T.conj().dot(hamiltonian.dot(op.dot(psi)))[0,0].real)
    #                grad2[i,j,l,k] = aa
    #                grad2[j,i,l,k] = -aa
    #                grad2[j,i,k,l] = aa
    #                grad2[i,j,k,l] = -aa
    #print(a[np.abs(a)>1e-8])
    #print(grad2[np.abs(grad2)>1e-8])
    #dgrad = a - grad2
    #print("\n grad difference")
    #print(dgrad[np.abs(dgrad)>1e-8])

    if spin_adapted==1:
        grad = general_to_spinadapted(a,pool,norb,aa)
    else: 
        grad = a

    return grad


def get_acse_slow_sa(psi,ham,pool):
    
    pool.generate_SparseMatrix()

    nlen = pool.n_ops
    a = np.zeros(nlen)
    for i in range(nlen):
        op = pool.spmat_ops[i]
        a[i] = (psi.T.conj().dot(op.dot(ham.dot(psi)))[0,0].real)
        print("xxx %12.10f" %a[i])
    return a


def get_acse_slow(psi,ham,nqubit):
    
    a = np.zeros((nqubit,nqubit,nqubit,nqubit))
    for i in range(nqubit):
        for j in range(i,nqubit):
            for k in range(nqubit):
                for l in range(k,nqubit):
                    op = get_2e_op(i,j,k,l,nqubit)
                    a[i,j,k,l] = (psi.T.conj().dot(op.dot(ham.dot(psi)))[0,0].real)
                    a[i,j,k,l]-= (psi.T.conj().dot(ham.dot(op.dot(psi)))[0,0].real)
                    a[i,j,l,k] = -a[i,j,k,l]
                    a[j,i,l,k] = a[i,j,k,l]
                    a[j,i,k,l] = -a[i,j,k,l]
    return a


def general_to_spinadapted(aa,pool,n_orb,aa2=None):

    grad = []

    # a0 = <\psi| [H, a_p^+ a_q^+ a_r a_s] |\psi>
    for op in pool.fermi_ops:
        a0 = 0
        for t in op.terms:
            coeff = op.terms[t]
            if len(t)==4:
                p = t[0][0]
                q = t[1][0]
                r = t[2][0]
                s = t[3][0]    
                a0 -= aa[p,q,r,s]*coeff 
            else:
                p = t[0][0]
                q = t[1][0]
                a0 -= aa2[p,q]*coeff
        grad.append(a0)

    return np.array(grad)

class ACSE:
    """
    compute anti-Hermitian Schrodinger equation
    <\psi|[a_p^+ a_q^+ a_r a_s,H]|\psi> = 0
    """
    def __init__(self):
        self.op_pool = []
        self.sq_ham = []
        self.ne = 0
        self.norb = 0
        self.nqubit = 0
        self.method = 0
 
    def init(self,sq_ham,pool,ne,norb,nqubit,method=1):
        self.op_pool = pool
        self.sq_ham = sq_ham
        self.ne = ne
        self.norb = norb
        self.nqubit = nqubit
        self.method = method

    def get_acse(self,psi):
        
        grad = get_acse(psi,self.sq_ham,self.op_pool,self.ne,self.norb,self.nqubit,self.method,spin_adapted=1)
        return grad


    def get_acse_v(self,psi):

        grad = get_acse(psi,self.sq_ham,self.op_pool,self.ne,self.norb,self.nqubit,method=1,spin_adapted=1)
        return grad

    def get_acse_ny(self,psi):

        grad = get_acse(psi,self.sq_ham,self.op_pool,self.ne,self.norb,self.nqubit,method=4,spin_adapted=1)
        return grad

