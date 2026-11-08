from openfermion import FermionOperator, QubitOperator
from openfermion.linalg import get_sparse_operator
from operators import Operators

class QubitOps(Operators):
    ''' Qubit operator
    '''

    def one_body_operator(self,p,q):
        '''
        '''
        pa = 2*p; pb = 2*p+1
        qa = 2*q; qb = 2*q+1
        self.one_body_operator_element(pa,qa)
        self.one_body_operator_element(pb,qb)

    def two_body_operator(self,p,q,r,s):
        '''
        '''
        pa = 2*p; pb = 2*p+1
        qa = 2*q; qb = 2*q+1
        ra = 2*r; rb = 2*r+1
        sa = 2*s; sb = 2*s+1
        if 1:
            self.two_body_operator_element(ra,sb,pa,qb)
            if s>r and q>p:
                self.two_body_operator_element(ra,sa,pa,qa)
                self.two_body_operator_element(rb,sb,pb,qb)
                self.two_body_operator_element(rb,sa,pa,qb)
            if q>p:
                self.two_body_operator_element(ra,sb,pb,qa)
            if s>r:
                self.two_body_operator_element(rb,sa,pb,qa)
        else:
            self.two_body_operator_element(ra,sb,pa,qb)
            self.two_body_operator_element(ra,sa,pa,qa)
            self.two_body_operator_element(rb,sb,pb,qb)
            self.two_body_operator_element(rb,sa,pa,qb)
            self.two_body_operator_element(ra,sb,pb,qa)
            self.two_body_operator_element(rb,sa,pb,qa)



    def one_body_operator_element(self,i,j):
        '''
        '''
        if self._isym=='IQEB':
            self.one_iqeb_operator(i,j) 
        elif self._isym=='QUBIT':
            self.one_qubit_operator(i,j)
        elif self._isym=='REDUCED_QUBIT':
            self.one_reduced_qubit_operator(i,j)
        else:
            print("Incorrect Qubit Operator type")
            exit()

    def two_body_operator_element(self,i,j,k,l):
        '''
        '''
        if self.two_identical_index(i,j,k,l):
            return

        if self._isym=='IQEB':
            self.two_iqeb_operator(i,j,k,l) 
        elif self._isym=='QUBIT':
            self.two_qubit_operator(i,j,k,l)
        elif self._isym=='REDUCED_QUBIT':
            self.two_reduced_qubit_operator(i,j,k,l)
        else:
            print("Incorrect Qubit Operator type")
            exit()

    def one_iqeb_operator(self,i,j):
        ''' One-body qubit excitation operator
            (Iterative qubit excitation operator arXiv:2011.10540v1) 
        '''
        if i>=j: return
        opi1 = QubitOperator((i,'X'),0.5)-QubitOperator((i,'Y'),1j/2)
        opj1 = QubitOperator((j,'X'),0.5)-QubitOperator((j,'Y'),1j/2)
        opi0 = QubitOperator((i,'X'),0.5)+QubitOperator((i,'Y'),1j/2)
        opj0 = QubitOperator((j,'X'),0.5)+QubitOperator((j,'Y'),1j/2)

        op = opi1*opj0 - opj1*opi0
        if op.many_body_order() > 0:
            self._ops.append(op)

        if self._complex:
            op = (opi1*opj0 + opj1*opi0)*1j
            if op.many_body_order() > 0:
                self._ops.append(op)


    def two_iqeb_operator(self,i,j,k,l):
        ''' Two-body qubit excitation operator
            (Iterative qubit excitation operator arXiv:2011.10540v1)
        '''
        opi1 = QubitOperator((i,'X'),0.5)-QubitOperator((i,'Y'),1j/2)
        opj1 = QubitOperator((j,'X'),0.5)-QubitOperator((j,'Y'),1j/2)
        opk1 = QubitOperator((k,'X'),0.5)-QubitOperator((k,'Y'),1j/2)
        opl1 = QubitOperator((l,'X'),0.5)-QubitOperator((l,'Y'),1j/2)
        opi0 = QubitOperator((i,'X'),0.5)+QubitOperator((i,'Y'),1j/2)
        opj0 = QubitOperator((j,'X'),0.5)+QubitOperator((j,'Y'),1j/2)
        opk0 = QubitOperator((k,'X'),0.5)+QubitOperator((k,'Y'),1j/2)
        opl0 = QubitOperator((l,'X'),0.5)+QubitOperator((l,'Y'),1j/2)

        op = opi1*opj1*opk0*opl0 - opl1*opk1*opj0*opi0
        if op.many_body_order() > 0:
            self._ops.append(op)

        if self._complex:
            op = (opi1*opj1*opk0*opl0 + opl1*opk1*opj0*opi0)*1j
            if op.many_body_order() > 0:
                self._ops.append(op)

    def one_qubit_operator(self,i,j):
        ''' One-body qubit operator
            (Qubit-ADAPT-VQE PRX QUANTUM 2, 020310)
        '''
        op = QubitOperator(((i,'X'),(j,'Y')),1j)
        if self.op_add(op): self._ops.append(op)
        op = QubitOperator(((i,'Y'),(j,'X')),1j)
        if self.op_add(op): self._ops.append(op)

        if self._complex:
            op = QubitOperator(((i,'X'),(j,'X')),1j)
            if self.op_add(op): self._ops.append(op)
            op = QubitOperator(((i,'Y'),(j,'Y')),1j)
            if self.op_add(op): self._ops.append(op)

    def two_qubit_operator(self,i,j,k,l):
        ''' Two-body qubit operator
            (Qubit-ADAPT-VQE PRX QUANTUM 2, 020310)
        '''
        op = QubitOperator(((i,'X'),(j,'Y'),(k,'X'),(l,'X')),1j)
        if self.op_add(op): self._ops.append(op)
        op = QubitOperator(((i,'Y'),(j,'X'),(k,'X'),(l,'X')),1j)
        if self.op_add(op): self._ops.append(op)
        op = QubitOperator(((i,'Y'),(j,'Y'),(k,'Y'),(l,'X')),1j)
        if self.op_add(op): self._ops.append(op)
        op = QubitOperator(((i,'Y'),(j,'Y'),(k,'X'),(l,'Y')),1j)
        if self.op_add(op): self._ops.append(op)
        op = QubitOperator(((i,'X'),(j,'X'),(k,'Y'),(l,'X')),1j)
        if self.op_add(op): self._ops.append(op)
        op = QubitOperator(((i,'X'),(j,'X'),(k,'X'),(l,'Y')),1j)
        if self.op_add(op): self._ops.append(op)
        op = QubitOperator(((i,'Y'),(j,'X'),(k,'Y'),(l,'Y')),1j)
        if self.op_add(op): self._ops.append(op)
        op = QubitOperator(((i,'X'),(j,'Y'),(k,'Y'),(l,'Y')),1j)
        if self.op_add(op): self._ops.append(op)

        if self._complex:
            op = QubitOperator(((i,'X'),(j,'X'),(k,'X'),(l,'X')),1j)
            if self.op_add(op): self._ops.append(op)
            op = QubitOperator(((i,'Y'),(j,'Y'),(k,'X'),(l,'X')),1j)
            if self.op_add(op): self._ops.append(op)
            op = QubitOperator(((i,'X'),(j,'Y'),(k,'Y'),(l,'X')),1j)
            if self.op_add(op): self._ops.append(op)
            op = QubitOperator(((i,'X'),(j,'Y'),(k,'X'),(l,'Y')),1j)
            if self.op_add(op): self._ops.append(op)
            op = QubitOperator(((i,'Y'),(j,'X'),(k,'Y'),(l,'X')),1j)
            if self.op_add(op): self._ops.append(op)
            op = QubitOperator(((i,'Y'),(j,'X'),(k,'X'),(l,'Y')),1j)
            if self.op_add(op): self._ops.append(op)
            op = QubitOperator(((i,'X'),(j,'X'),(k,'Y'),(l,'Y')),1j)
            if self.op_add(op): self._ops.append(op)
            op = QubitOperator(((i,'Y'),(j,'Y'),(k,'Y'),(l,'Y')),1j)
            if self.op_add(op): self._ops.append(op)

    def one_reduced_qubit_operator(self,i,j):
        ''' One-body qubit operator
            (Qubit-ADAPT-VQE PRX QUANTUM 2, 020310)
        '''
        op = QubitOperator(((i,'X'),(j,'Y')),1j)
        if op.many_body_order() > 0 and self.op_add(op):
            self._ops.append(op)

        op = QubitOperator(((i,'Y'),(j,'X')),1j)
        if op.many_body_order() > 0 and self.op_add(op):
            self._ops.append(op)


    def two_reduced_qubit_operator(self,i,j,k,l):
        ''' Two-body qubit operator
            (Qubit-ADAPT-VQE PRX QUANTUM 2, 020310)
        '''
        op = QubitOperator(((i,'X'),(j,'Y'),(k,'X'),(l,'X')),1j)
        if op.many_body_order() > 0 and self.op_add(op):
            self._ops.append(op)

        op = QubitOperator(((i,'X'),(j,'X'),(k,'Y'),(l,'X')),1j)
        if op.many_body_order() > 0 and self.op_add(op):
            self._ops.append(op)


    def op_add(self,op):
        '''
        ''' 
        to_add = False; not_x_and_y = False
        nx=0; ny=0
        for t in op.terms:
            for ti in t:
                if ti[1] == 'X': nx += 1
                elif ti[1] == 'Y': ny += 1
                else: not_x_and_y = True

        if (nx+ny) >= 2:
            to_add = True
            for i in range(len(self._ops)):
                if op==self._ops[i]:
                    to_add = False
                    break

        to_add = op.many_body_order() > 0 and to_add

        return to_add

