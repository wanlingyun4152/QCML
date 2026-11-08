import pyscf
from pyscf import gto, scf, data
import numpy as np
from integrators import VelocityVerlet

def test(r):
    mol = gto.Mole()
    mol.atom = [('Li', (0,0,0)), ('H', (0,0,r))]
    mol.charge = 0
    mol.spin = 0
    mol.basis = 'sto-3g'
    mol.symmetry = 1
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    mycas = mf.CASCI(6,4)
    myscanner = mycas.nuc_grad_method().as_scanner()
#    etot, grad = myscanner(mol)
#    print("forces:", grad, "a.u.")

    mass = []
    veloc = []
    T = 300 #K
    for c in mol.atom_charges():
        m = data.elements.COMMON_ISOTOPE_MASSES[c] * data.nist.AMU2AU
        v = np.sqrt(3*(data.nist.BOLTZMANN / data.nist.HARTREE2J) * T / m)
        mass.append(m)
        veloc.append(v)
    init_veloc = np.array([[0,0,veloc[0]],
                           [0,0,veloc[1]]])
    print("init_veloc =", init_veloc, "a.u.")

    myintegrator = VelocityVerlet(myscanner,
                                dt=5,
                                steps=2, #85000,
                                veloc=init_veloc,
                                data_output=f"BOMD_{r}.md.data",
                                trajectory_output=f"BOMD_{r}.md.xyz").run()

if __name__== "__main__":

    r = 1.5475 #fci 1.5475
#    print("System LiH", "Bond Length =", r)
    test(r)


