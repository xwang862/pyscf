#!/usr/bin/env python
import numpy as np
from pyscf.pbc import gto, scf, cc
from pyscf.pbc.cc import eom_kccsd_rhf as eom_krccsd
from pyscf.pbc.tools import pyscf_ase
from pyscf.pbc.tools import lattice

##############################
# Create a "Cell"
##############################
cell = gto.Cell()
cell.unit = 'B'
formula = 'lif'
ase_atom = lattice.get_ase_atom(formula)
cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
cell.a = ase_atom.cell
cell.basis = 'gth-szv'
cell.pseudo = "gth-pade"
cell.verbose = 5
cell.max_memory = 4000 # MB

cell.build()

##############################
#  KRHF with Gaussian density fitting (GDF)
##############################

# number of kpts for each axis
nk = 2
nmp = [nk, nk, nk]
kpts = cell.make_kpts(nmp)

# set up k-point scf (khf)
mymf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
mymf = mymf.density_fit()

# run SCF
ekrhf = mymf.kernel()

##############################
# PBC K-point closed-shell CCSD: KRCCSD
##############################

mycc = cc.KRCCSD(mymf, frozen=0)
mycc.keep_exxdiv = True

eris = mycc.ao2mo()
ekrcc, t1, t2 = mycc.kernel(eris=eris)

##############################
# KRCCSD spectra
##############################
myeom = eom_krccsd.EOMEESinglet(mycc)

wmin = 0.33
wmax = 0.73
dw = 0.005
eta = 0.005

nw = int((wmax - wmin) / dw) + 1
scan = np.linspace(wmin, wmax, nw)

spectrum, sol = myeom.get_absorption_spectrum(scan, eta, eris=eris, tol=1e-1, maxiter=10)
spectrum = np.sum(spectrum, axis=0)

np.savetxt("spectrum.txt", np.column_stack([scan,spectrum]))
