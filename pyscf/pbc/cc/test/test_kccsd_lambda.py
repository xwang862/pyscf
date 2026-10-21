#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Lambda equations of the k-point CCSD codes on multi-k meshes, KRCCSD (kccsd_rhf_lambda) and KGCCSD
(kccsd_lambda). Lambda is checked through quantities that do not depend on the k-point Lambda solver:
  * orbital-invariant norms against the equivalent supercell at the Gamma point;
  * invariance under arbitrary phases of the Bloch orbitals;
  * KGCCSD against the spin-orbital form of KRCCSD.
H2O in a small cubic box has no inversion center, so all blocks of Lambda contribute.
'''

import unittest
import numpy as np
from pyscf.pbc import gto, scf, cc
from pyscf.pbc.tools.pbc import super_cell


def setUpModule():
    global cell
    r, th = 0.9572, np.deg2rad(104.52)
    cell = gto.Cell()
    cell.atom = [['O', (0., 0., 0.)],
                 ['H', (r * np.cos(th / 2), r * np.sin(th / 2), 0.)],
                 ['H', (r * np.cos(th / 2), -r * np.sin(th / 2), 0.)]]
    cell.a = np.eye(3) * 3.19
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.build()

def tearDownModule():
    global cell
    del cell


def run_krhf(kpts, exxdiv=None):
    kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv).density_fit()
    kmf.conv_tol = 1e-11
    kmf.kernel()
    return kmf

def run_lambda(mycc):
    mycc.conv_tol = 1e-11
    mycc.conv_tol_normt = 1e-9
    mycc.kernel()
    mycc.solve_lambda()
    return mycc


class KnownValues(unittest.TestCase):
    def test_krccsd_lambda_vs_supercell(self):
        nmp = [2, 1, 1]
        kcc = run_lambda(cc.KRCCSD(run_krhf(cell.make_kpts(nmp))))

        mf = scf.RHF(super_cell(cell, nmp), exxdiv=None).density_fit()
        mf.conv_tol = 1e-11
        mf.kernel()
        mycc = run_lambda(cc.RCCSD(mf))

        self.assertAlmostEqual(kcc.e_corr, mycc.e_corr / 2, 8)
        self.assertAlmostEqual(np.linalg.norm(kcc.t2), np.linalg.norm(mycc.t2), 7)
        self.assertAlmostEqual(np.linalg.norm(kcc.l1), np.linalg.norm(mycc.l1), 6)
        self.assertAlmostEqual(np.linalg.norm(kcc.l2), np.linalg.norm(mycc.l2), 7)

    def test_krccsd_lambda_phase_invariance(self):
        # Shifted mesh (k and -k are not both present), Madelung-shifted denominators (exxdiv='ewald')
        kmf = run_krhf(cell.make_kpts([3, 1, 1], scaled_center=[0.1, 0., 0.]), exxdiv='ewald')
        ref = run_lambda(cc.KRCCSD(kmf))
        rng = np.random.default_rng(1)
        mo_coeff = [c * np.exp(1j * rng.uniform(0, 2 * np.pi, c.shape[1])) for c in kmf.mo_coeff]
        mycc = run_lambda(cc.KRCCSD(kmf, mo_coeff=mo_coeff))

        self.assertAlmostEqual(mycc.e_corr, ref.e_corr, 9)
        self.assertAlmostEqual(np.linalg.norm(mycc.l1), np.linalg.norm(ref.l1), 8)
        self.assertAlmostEqual(np.linalg.norm(mycc.l2), np.linalg.norm(ref.l2), 8)

    def test_kgccsd_lambda_vs_krccsd(self):
        kmf = run_krhf(cell.make_kpts([3, 1, 1], scaled_center=[0.1, 0., 0.]))
        rcc = run_lambda(cc.KRCCSD(kmf))
        gcc = run_lambda(cc.KGCCSD(kmf))
        orbspin = getattr(gcc.ao2mo(), 'orbspin', None)

        # the spin-orbital mapping is checked on T first
        t2 = gcc.spatial2spin(np.asarray(rcc.t2), orbspin)
        self.assertAlmostEqual(abs(np.asarray(gcc.t2) - t2).max(), 0, 8)
        l1 = gcc.spatial2spin(np.asarray(rcc.l1), orbspin)
        l2 = gcc.spatial2spin(np.asarray(rcc.l2), orbspin)
        self.assertAlmostEqual(abs(np.asarray(gcc.l1) - l1).max(), 0, 8)
        self.assertAlmostEqual(abs(np.asarray(gcc.l2) - l2).max(), 0, 8)


if __name__ == '__main__':
    print('Lambda equations of the k-point CCSD codes')
    unittest.main()
