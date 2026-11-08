"""Exponential-of-sum UCC; k-UpCCGSD is an ORDERED product of layer exponentials.
Gradients use converged Gauss-Legendre integration of the Frechet derivative.
No approximation [G,H]@psi is substituted for a noncommuting exponential derivative.
"""
import numpy as np
from scipy.sparse.linalg import expm_multiply
from numpy.polynomial.legendre import leggauss

def evolve(A,v,scale=1.):
    if scale==0:return np.asarray(v).copy()
    return expm_multiply(A*float(scale),v,traceA=0.)

class Ansatz:
    def __init__(self,compiled,hamiltonian,layers=1):
        if not 1<=layers<=3:raise ValueError('layers must be 1..3')
        self.pool=compiled;self.h=hamiltonian;self.layers=layers;self.nparams=layers*compiled.nops
        self.last_quadrature={}
    def _forward(self,theta):
        t=np.asarray(theta,dtype=float)
        if t.shape!=(self.nparams,) or not np.isfinite(t).all():raise ValueError('Bad theta')
        mats=[self.pool.combine(q) for q in t.reshape(self.layers,-1)]
        states=[self.pool.sector.hf()]
        for A in mats:states.append(evolve(A,states[-1]))
        return mats,states
    def state(self,t):return self._forward(t)[1][-1]
    def energy(self,t):
        p=self.state(t);return float(np.dot(p,self.h.apply(p))/np.dot(p,p))
    def _integrate(self,A,phi,adj,order):
        x,w=leggauss(order);x=(x+1)/2;w=w/2
        r=evolve(A,phi,x[0]);l=evolve(A,adj,x[0]-1);g=np.zeros(self.pool.nops)
        for i in range(order):
            if i:
                delta=x[i]-x[i-1];r=evolve(A,r,delta);l=evolve(A,l,delta)
            g+=2*w[i]*self.pool.contractions(l,r)
        return g
    def value_grad(self,theta):
        mats,states=self._forward(theta);p=states[-1];n=float(np.dot(p,p));hp=self.h.apply(p);e=float(np.dot(p,hp)/n)
        if abs(n-1)>1e-8:raise RuntimeError('Unitary norm check failed')
        adj=(hp-e*p)/n;grad=[];orders=[];errors=[]
        for A,phi in reversed(list(zip(mats,states[:-1]))):
            low=self._integrate(A,phi,adj,8)
            for order in (16,32,64):
                high=self._integrate(A,phi,adj,order)
                err=float(abs(high-low).max(initial=0.))
                if np.all(abs(high-low)<=1e-9+1e-7*abs(high)):
                    grad.append(high);orders.append(order);errors.append(err);break
                low=high
            else:raise RuntimeError('Gradient quadrature failed at order64; no forced convergence')
            adj=evolve(A,adj,-1.)
        self.last_quadrature={'orders_last_to_first':orders,'difference_max_last_to_first':errors,'definition':'successive Gauss-Legendre orders; numerical estimator, not rigorous error bound'}
        return e,np.concatenate(grad[::-1])
