"""Real fermion polynomial algebra. Interleaved labels: 2p=alpha, 2p+1=beta.

Normal order: creators left, decreasing mode labels within each action block.
The coefficient L2 normalization matches the uploaded FermionOps definition,
not a Hilbert-space Frobenius normalization. No qubit truncation or zero-label deletion.
"""
from functools import lru_cache
import math
import numbers

@lru_cache(maxsize=200000)
def _normal_word(word):
    # Adjacent canonical anticommutation, including contraction when a_i a_i^+.
    for i in range(len(word)-1):
        p, ap = word[i]; q, aq = word[i+1]
        if ap == aq and p == q:
            return ()
        swap = (ap < aq) or (ap == aq and p < q)
        if swap:
            out = {}
            swapped = word[:i]+(word[i+1],word[i])+word[i+2:]
            for t, c in _normal_word(swapped): out[t] = out.get(t,0)-c
            if ap == 0 and aq == 1 and p == q:
                for t,c in _normal_word(word[:i]+word[i+2:]): out[t] = out.get(t,0)+c
            return tuple((t,c) for t,c in sorted(out.items()) if c)
    return ((word,1),)

class Fermion:
    """Minimal polynomial, including the operators used in the original source."""
    def __init__(self, term=None, coefficient=1.):
        self.terms = {} if term is None else {tuple(tuple(x) for x in term): coefficient}
    @classmethod
    def from_dict(cls,d):
        obj=cls();obj.terms={k:v for k,v in d.items() if v != 0};return obj
    def __add__(self,other):
        if other == 0 if isinstance(other,numbers.Number) else False: return Fermion.from_dict(self.terms)
        if not isinstance(other,Fermion): return NotImplemented
        d=self.terms.copy()
        for t,c in other.terms.items():d[t]=d.get(t,0)+c
        return Fermion.from_dict(d)
    __radd__=__add__
    def __neg__(self):return self*-1
    def __sub__(self,o):return self+-o
    def __mul__(self,o):
        if isinstance(o,numbers.Number):return Fermion.from_dict({t:c*o for t,c in self.terms.items()})
        if not isinstance(o,Fermion):return NotImplemented
        d={}
        for t,c in self.terms.items():
            for u,v in o.terms.items():d[t+u]=d.get(t+u,0)+c*v
        return Fermion.from_dict(d)
    __rmul__=__mul__
    def __truediv__(self,v):return self*(1/v)
    def many_body_order(self):return max((len(t) for t,c in self.terms.items() if abs(c)>1e-14),default=0)

def normal_ordered(poly):
    out={}
    for word,c in poly.terms.items():
        for t,s in _normal_word(word):out[t]=out.get(t,0)+c*s
    return Fermion.from_dict({t:c for t,c in out.items() if abs(c)>1e-14})

def dagger(poly):
    return Fermion.from_dict({tuple((p,1-a) for p,a in reversed(t)):complex(c).conjugate() for t,c in poly.terms.items()})

def normalized_anti(poly):
    out=normal_ordered(poly-dagger(poly))
    if out.many_body_order()==0:return None
    if any(abs(complex(c).imag)>1e-14 for c in out.terms.values()):raise ValueError('Real pool only')
    d={t:float(complex(c).real) for t,c in out.terms.items()}
    scale=math.sqrt(sum(c*c for c in d.values()))
    if scale == 0: return None
    return Fermion.from_dict({t:c/scale for t,c in d.items()})

def raw_single(p,q):return Fermion(((2*p,1),(2*q,0)))+Fermion(((2*p+1,1),(2*q+1,0)))

def raw_sa_doubles(p,q,r,s):
    pa,pb,qa,qb,ra,rb,sa,sb=2*p,2*p+1,2*q,2*q+1,2*r,2*r+1,2*s,2*s+1
    A=Fermion(((pa,1),(qa,1),(ra,0),(sa,0)))+Fermion(((pb,1),(qb,1),(rb,0),(sb,0)))
    C=Fermion(((pa,1),(qb,1),(ra,0),(sb,0)))+Fermion(((pb,1),(qa,1),(rb,0),(sa,0)))
    E=Fermion(((pb,1),(qa,1),(ra,0),(sb,0)))+Fermion(((pa,1),(qb,1),(rb,0),(sa,0)))
    return [2*A+C+E,C-E]

def pool(norb,nocc,family):
    if not 1<=nocc<norb<=19:raise ValueError('Require 1<=nocc<norb<=19')
    if family not in ('uccsd','uccgsd','kupccgsd'):raise ValueError('Unknown family')
    if family=='uccgsd' and norb>8:raise ValueError('Full generalized SD pilot limited to m<=8; no silent truncation')
    out=[];labels=[]
    def add(raw,label):
        op=normalized_anti(raw)
        if op is not None:out.append(op);labels.append(label)
    if family=='uccsd':
        for i in range(nocc):
            for a in range(nocc,norb):add(raw_single(a,i),['single',a,i])
        for i in range(nocc):
            for j in range(i,nocc):
                for a in range(nocc,norb):
                    for b in range(a,norb):
                        for spin,raw in enumerate(raw_sa_doubles(a,b,i,j)):add(raw,['double_sa',a,b,i,j,spin])
    else:
        for p in range(norb):
            for q in range(p,norb):add(raw_single(p,q),['single',p,q])
        if family=='kupccgsd':
            for p in range(norb):
                for q in range(p,norb):
                    for spin,raw in enumerate(raw_sa_doubles(p,p,q,q)):add(raw,['pair_sa',p,q,spin])
        else:
            pairs=[(p,q) for p in range(norb) for q in range(p,norb)]
            for pq,(p,q) in enumerate(pairs):
                for rs in range(pq,len(pairs)):
                    r,s=pairs[rs]
                    for spin,raw in enumerate(raw_sa_doubles(p,q,r,s)):add(raw,['double_sa',p,q,r,s,spin])
    return out,labels
