import pyscf
from pyscf import gto, scf, fci
import os
import numpy

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

    mo_energy = mf.mo_energy
    e_idx = numpy.argsort(mo_energy)
    e_sort = mo_energy[e_idx]
    nmo = mo_energy.size
    mo_occ = numpy.zeros_like(mo_energy)
    nocc = mf.mol.nelectron // 2
    mo_occ[e_idx[:nocc]] = 2
    homo = round(e_sort[nocc-1]*27.2114,10)
    lumo = round(e_sort[nocc]*27.2114,10)

    myfci = fci.FCI(mf)
    myfci.kernel()
    fci_e = myfci.e_tot

    return homo, lumo, fci_e

#if __name__== "__main__":
def input_f(r):

    filename = "nn_input.txt"
    os.system("rm -rf %s" % filename)
    zeroindex = numpy.loadtxt("zeroindex_LiH.txt")
    title_list = ["system", "bondlengths", "no_qubits", "no_parameters",
                  "no_gates", "no_Pauli_strings", "no_electron",
                  "HOMO (eV)", "LUMO (eV)", "FCI_energy", "VQE_energy"
                  ] + [f"VQE_param{i}" for i in range(1,331) if i not in zeroindex]
    with open(filename,"a") as f:
        for title_i in range(len(title_list)):
            if title_i < len(title_list)-1:
                f.write(title_list[title_i] + "\t")
            else:
                f.write(title_list[title_i] + "\n")

    #r = 1.5475
#    print("System LiH", "Bond Length =", r)
    homo, lumo, fci_e = test(r)

    value_list = ["LiH", f"{r}", "12", "330",
                  "61484", "631", "4",
                  f"{homo}", f"{lumo}", f"{fci_e}", f"{fci_e}"
                  ] + [f"{j}" for j in range(1,331) if j not in zeroindex]

    with open(filename,"a") as f:
        for title_i in range(len(title_list)):
            if title_i < len(title_list)-1:
                f.write(value_list[title_i] + "\t")
            else:
                f.write(value_list[title_i] + "\n")


