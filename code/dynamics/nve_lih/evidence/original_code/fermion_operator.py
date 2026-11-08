from openfermion import FermionOperator, hermitian_conjugated, normal_ordered 
from openfermion.linalg import get_sparse_operator
from operators import Operators
import numpy as np

class FermionOps(Operators):
    ''' fermion operator
    '''
    def anti_hermitian_operator(self, termA, itype=0):
        ''' Build Anti-Hermitian Operator
        args:
            termA: general Fermionic operator
            itype: type of Anti-Hermitian Operator 
                0: (termA - h.c.)
                1: (termA + h.c.)*1j
        '''
        if itype==0:
            termB = termA - hermitian_conjugated(termA)
        else:
            termB = (termA + hermitian_conjugated(termA))

        termB = normal_ordered(termB)

        coeffB = 0
        for t in termB.terms:
            coeff_t = termB.terms[t]
            coeffB += coeff_t * coeff_t

        if termB.many_body_order() > 0:
            termB = termB/np.sqrt(coeffB)
            if itype==1:
                termB *= 1j
            self._ops.append(termB)
    
        return

    def one_body_operator(self,i,j):
        ''' One body fermionic operators
        isym: symmetry of fermionic operators
          = 0: general anti-Hermitian operators
          = 1: spin-complete anti-Hermitian operators
          = 2: spin-adapted anti-Hermitian operators
        '''
        isym = self._isym.lower()
        if isym=='sc' or isym=='sa':
            termA  = FermionOperator(((2*i,1),(2*j,0)))
            termA += FermionOperator(((2*i+1,1),(2*j+1,0)))
            self.anti_hermitian_operator(termA)
            if self._complex: 
                self.anti_hermitian_operator(termA,1)
        elif isym=='general':
            termA = FermionOperator(((2*i,1),(2*j,0)))
            termB = FermionOperator(((2*i+1,1),(2*j+1,0)))
            self.anti_hermitian_operator(termA) 
            self.anti_hermitian_operator(termB)
            if self._complex:
                self.anti_hermitian_operator(termA,1)
                self.anti_hermitian_operator(termB,1)
        else:
            print('Chech Fermionic Operator Symmetry')
            exit()

    def sc_two_body_operator_element(self, p, q, r, s, termA, termB, 
                                     termC, icomplex):
        '''
        ''' 
        if q>p and s>r:
            self.anti_hermitian_operator(termA,icomplex)

        self.anti_hermitian_operator(termB,icomplex)

        if q>p or s>r:
            self.anti_hermitian_operator(termC,icomplex)

    def general_two_body_operator_element(self, p, q, r, s, termA, termB,
                                 termC, termD, termE, termF, add_termA, add_termB, 
                                 add_termC, add_termD, add_termE, add_termF, icomplex):
        '''
        '''
        if add_termC:
            self.anti_hermitian_operator(termC,icomplex)
        if s>r and q>p:
            if add_termA:
                self.anti_hermitian_operator(termA,icomplex)
            if add_termB:
                self.anti_hermitian_operator(termB,icomplex)
            if add_termD:
                self.anti_hermitian_operator(termD,icomplex)
        if q>p:
            if add_termE:
                self.anti_hermitian_operator(termE,icomplex)
        if s>r:
            if add_termF:
                self.anti_hermitian_operator(termF,icomplex)
 

    def two_body_operator(self,p,q,r,s):
        ''' Two body fermionic operator
        isym: symmetry of fermionic operators
          = 0: general anti-Hermitian operators
          = 1: spin-complete anti-Hermitian operators
          = 2: spin-adapted anti-Hermitian operators
        '''
        pa = 2*p; pb = 2*p+1
        qa = 2*q; qb = 2*q+1
        ra = 2*r; rb = 2*r+1
        sa = 2*s; sb = 2*s+1

        termA = FermionOperator(((pa,1),(qa,1),(ra,0),(sa,0)))
        termB = FermionOperator(((pb,1),(qb,1),(rb,0),(sb,0)))
        termC = FermionOperator(((pa,1),(qb,1),(ra,0),(sb,0)))
        termD = FermionOperator(((pb,1),(qa,1),(rb,0),(sa,0)))
        termE = FermionOperator(((pb,1),(qa,1),(ra,0),(sb,0)))
        termF = FermionOperator(((pa,1),(qb,1),(rb,0),(sa,0)))

        add_termA = not self.two_identical_index(pa,qa,ra,sa)
        add_termB = not self.two_identical_index(pb,qb,rb,sb)
        add_termC = not self.two_identical_index(pa,qb,ra,sb)
        add_termD = not self.two_identical_index(pb,qa,rb,sa)
        add_termE = not self.two_identical_index(pb,qa,ra,sb)
        add_termF = not self.two_identical_index(pa,qb,rb,sa)

        isym = self._isym.lower()
        if isym=='sc':
            termA += termB
            termC += termD
            termE += termF
            self.sc_two_body_operator_element(p,q,r,s,termA,termC,termE,0)
            
            if self._complex:
                self.sc_two_body_operator_element(p,q,r,s,termA,termC,termE,1)
                
        elif isym=='sa':
            termA += termB
            termC += termD
            termE += termF

            self.anti_hermitian_operator(termA*2+termC+termE)
            self.anti_hermitian_operator(termC-termE)
            
            if self._complex:
                self.anti_hermitian_operator(termA*2+termC+termE,1)
                self.anti_hermitian_operator(termC-termE,1)
 
        elif isym=='general':
            self.general_two_body_operator_element(p,q,r,s,termA,
                                  termB,termC,termD,termE,termF,add_termA,add_termB,
                                  add_termC,add_termD,add_termE,add_termF,0)
            if self._complex:
                self.general_two_body_operator_element(p,q,r,s,termA,
                                  termB,termC,termD,termE,termF,add_termA,add_termB,
                                  add_termC,add_termD,add_termE,add_termF,0)
            '''
            if 0:
                self.anti_hermitian_operator(termC)
                self.anti_hermitian_operator(termA)
                self.anti_hermitian_operator(termB)
                self.anti_hermitian_operator(termE)
                self.anti_hermitian_operator(termD)
                self.anti_hermitian_operator(termF)
                if self._complex:
                    self.anti_hermitian_operator(termC,1)
                    self.anti_hermitian_operator(termA,1)
                    self.anti_hermitian_operator(termB,1)
                    self.anti_hermitian_operator(termE,1)
                    self.anti_hermitian_operator(termD,1)
                    self.anti_hermitian_operator(termF,1)
            else:
                if add_termC:
                    self.anti_hermitian_operator(termC)
                if s>r and q>p:
                    if add_termA:
                        self.anti_hermitian_operator(termA)
                    if add_termB:
                        self.anti_hermitian_operator(termB)
                    if add_termD:
                        self.anti_hermitian_operator(termD)
                if q>p:
                    if add_termE:
                        self.anti_hermitian_operator(termE)
                if s>r:
                    if add_termF:
                        self.anti_hermitian_operator(termF)
                
                if self._complex:
                    self.anti_hermitian_operator(termC,1)
                    if s>r and q>p:
                        self.anti_hermitian_operator(termA,1)
                        self.anti_hermitian_operator(termB,1)
                        self.anti_hermitian_operator(termD,1)
                    if s>r:
                        self.anti_hermitian_operator(termF,1)
                    if q>p:
                        self.anti_hermitian_operator(termE,1)
            '''
        else:
            print('Chech Fermionic Operator Symmetry')
            exit()
