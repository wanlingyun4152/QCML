"""Fixed (Nalpha,Nbeta) fermion matrices, in PySCF alpha-major CI ordering."""
import math
import itertools
import numpy as np
from scipy import sparse
from .algebra import Fermion

_BYTE_POP=np.array([bin(i).count('1') for i in range(256)],dtype=np.uint8)

def popcounts(x):
    x=np.ascontiguousarray(x,dtype=np.uint64)
    return _BYTE_POP[x.view(np.uint8).reshape(-1,8)].sum(axis=1,dtype=np.int16)

class Sector:
    def __init__(self,norb,nalpha=2,nbeta=2):
        if not 1<=norb<=19 or not 0<=nalpha<=norb or not 0<=nbeta<=norb:raise ValueError('Invalid sector')
        self.norb,self.nalpha,self.nbeta=norb,nalpha,nbeta
        self.alpha=np.array(sorted(sum(1<<p for p in c) for c in itertools.combinations(range(norb),nalpha)),dtype=np.uint64)
        self.beta=np.array(sorted(sum(1<<p for p in c) for c in itertools.combinations(range(norb),nbeta)),dtype=np.uint64)
        self.shape=(len(self.alpha),len(self.beta));self.dimension=math.prod(self.shape)
        self.states=(self.alpha[:,None] | (self.beta[None,:]<<np.uint64(norb))).reshape(-1)
        self.sorted_indices=np.argsort(self.states);self.sorted_states=self.states[self.sorted_indices]
    def mode(self,interleaved):
        q=int(interleaved)
        if not 0<=q<2*self.norb:raise ValueError('Operator index outside orbitals')
        return q//2+(self.norb if q%2 else 0)
    def hf(self):
        x=np.zeros(self.dimension);x[0]=1.;return x
    def matrix(self,poly):
        rows=[];cols=[];vals=[]
        for word,coeff in poly.terms.items():
            if abs(complex(coeff).imag)>1e-14:raise ValueError('Real operators only')
            changes=[0,0]
            for q,a in word:changes[q%2]+=2*a-1
            if changes != [0,0]:raise ValueError('Operator not Nalpha/Nbeta conserving; projection is forbidden')
            st=self.states.copy();ids=np.arange(self.dimension,dtype=np.int32);sg=np.ones(self.dimension)
            for q,a in reversed(word):
                mode=self.mode(q);bit=np.uint64(1)<<np.uint64(mode)
                active=((st&bit)==0) if a else ((st&bit)!=0)
                st=st[active];ids=ids[active];sg=sg[active]
                if not len(st):break
                parity=popcounts(st&(bit-np.uint64(1)))&1
                sg*=1-2*parity
                st ^= bit
            if len(st):
                pos=np.searchsorted(self.sorted_states,st)
                if np.any(pos>=self.dimension) or not np.array_equal(self.sorted_states[pos],st):raise RuntimeError('Sector leakage')
                rows.append(self.sorted_indices[pos].astype(np.int32));cols.append(ids);vals.append(sg*float(complex(coeff).real))
        if not rows:return sparse.csr_matrix((self.dimension,self.dimension),dtype=float)
        mat=sparse.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(self.dimension,)*2).tocsr()
        mat.eliminate_zeros();mat.sort_indices();return mat
    def full_embedding(self):
        """Indices & signs mapping alpha-major CI to interleaved full-Fock JW (MSB mode0)."""
        if self.norb>6:raise ValueError('Full-Fock embedding is a small-system validation only')
        indices=[];signs=[]
        for a in self.alpha:
            for b in self.beta:
                al=[p for p in range(self.norb) if int(a)&(1<<p)]
                be=[p for p in range(self.norb) if int(b)&(1<<p)]
                count=sum(q<p for p in al for q in be)
                idx=sum(1<<(2*self.norb-1-2*p) for p in al)+sum(1<<(2*self.norb-2-2*q) for q in be)
                indices.append(idx);signs.append((-1.)**count)
        return np.array(indices),np.array(signs)

class CompiledPool:
    def __init__(self,sector,ops,max_entries=20000000):
        self.sector=sector;self.nops=len(ops);self.matrices=[]
        rows=[];cols=[];data=[];which=[];n=0
        for j,op in enumerate(ops):
            A=sector.matrix(op)
            if A.nnz and abs((A+A.T).data).max(initial=0)>1e-13:raise RuntimeError('Generator not antisymmetric')
            self.matrices.append(A);c=A.tocoo();n+=A.nnz
            if n>max_entries:raise MemoryError('Pool entry cap exceeded, not truncating')
            rows.append(c.row.astype(np.int32));cols.append(c.col.astype(np.int32));data.append(c.data);which.append(np.full(A.nnz,j,dtype=np.int32))
        self.rows=np.concatenate(rows);self.cols=np.concatenate(cols);self.data=np.concatenate(data);self.which=np.concatenate(which)
        self.entries=n
    def combine(self,theta):
        x=np.asarray(theta,dtype=float)
        if x.shape!=(self.nops,) or not np.isfinite(x).all():raise ValueError('Invalid parameter vector')
        out=sparse.coo_matrix((self.data*x[self.which],(self.rows,self.cols)),shape=(self.sector.dimension,)*2).tocsr()
        out.eliminate_zeros();return out
    def contractions(self,left,right):
        v=left[self.rows]*self.data*right[self.cols]
        return np.bincount(self.which,weights=v,minlength=self.nops)

class PySCFHamiltonian:
    def __init__(self,sector,h1,eri,enuc):
        from pyscf import fci
        self.s=sector;self.fci=fci;self.h1=np.ascontiguousarray(h1,dtype=float);self.eri=np.ascontiguousarray(eri,dtype=float);self.enuc=float(enuc)
        self.nelec=(sector.nalpha,sector.nbeta)
        self.h2=fci.direct_spin1.absorb_h1e(self.h1,self.eri,sector.norb,self.nelec,.5)
        self.links=fci.cistring.gen_linkstr_index_trilidx(range(sector.norb),sector.nalpha),fci.cistring.gen_linkstr_index_trilidx(range(sector.norb),sector.nbeta)
        if not np.array_equal(fci.cistring.make_strings(range(sector.norb),sector.nalpha),sector.alpha):raise RuntimeError('PySCF alpha ordering mismatch')
        if not np.array_equal(fci.cistring.make_strings(range(sector.norb),sector.nbeta),sector.beta):raise RuntimeError('PySCF beta ordering mismatch')
    def apply(self,v):
        x=np.ascontiguousarray(v,dtype=float).reshape(self.s.shape)
        y=self.fci.direct_spin1.contract_2e(self.h2,x,self.s.norb,self.nelec,link_index=self.links)
        return np.asarray(y).reshape(-1)+self.enuc*x.reshape(-1)
    def spin_square(self,v):
        return float(self.fci.spin_op.spin_square(np.asarray(v).reshape(self.s.shape),self.s.norb,self.nelec)[0])
