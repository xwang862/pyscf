#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import unittest
from pyscf import scf
from pyscf import gto
from pyscf import mcscf

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        ["C", (-0.65830719,  0.61123287, -0.00800148)],
        ["C", ( 0.73685281,  0.61123287, -0.00800148)],
        ["C", ( 1.43439081,  1.81898387, -0.00800148)],
        ["C", ( 0.73673681,  3.02749287, -0.00920048)],
        ["C", (-0.65808819,  3.02741487, -0.00967948)],
        ["C", (-1.35568919,  1.81920887, -0.00868348)],
        ["H", (-1.20806619, -0.34108413, -0.00755148)],
        ["H", ( 1.28636081, -0.34128013, -0.00668648)],
        ["H", ( 2.53407081,  1.81906387, -0.00736748)],
        ["H", ( 1.28693681,  3.97963587, -0.00925948)],
        ["H", (-1.20821019,  3.97969587, -0.01063248)],
        ["H", (-2.45529319,  1.81939187, -0.00886348)],]

    mol.basis = {'H': '6-31g',
                 'C': '6-31g',}
    mol.max_memory = 20
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()
    mo = mf.mo_coeff.copy()
    mo[:,[15,16,17,18]] = mf.mo_coeff[:,[17,18,15,16]]

def tearDownModule():
    global mol, mf
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_mc2step_4o4e(self):
        mc = mcscf.approx_hessian(mcscf.CASSCF(mf, 4, 4), auxbasis='weigend')
        mc.conv_tol = 1e-8
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -230.6627383822, 7)

    def test_mc2step_4o4e_df(self):
        mc = mcscf.CASSCF(mf, 4, 4).density_fit(auxbasis='weigend')
        mc.conv_tol = 1e-8
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -230.6617475545396, 7)

        mf1 = mf.density_fit(auxbasis='weigend')
        mc = mcscf.CASSCF(mf1, 4, 4).density_fit(auxbasis='weigend')
        mc.conv_tol = 1e-8
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -230.6617475545396, 7)

    def test_mc2step_9o8e(self):
        mc = mcscf.approx_hessian(mcscf.CASSCF(mf, 9, 8), auxbasis='weigend')
        mc.conv_tol = 1e-8
        mo = mf.mo_coeff.copy()
        mo[:,[15,16,17,18]] = mf.mo_coeff[:,[17,18,15,16]]
        emc = mc.mc2step(mo)[0]
        self.assertAlmostEqual(emc, -230.72211519779304, 5)

    def test_mc1step_4o4e(self):
        mc = mcscf.approx_hessian(mcscf.CASSCF(mf, 4, 4), auxbasis='weigend')
        mc.conv_tol = 1e-8
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -230.6627383823, 7)
        #?self.assertAlmostEqual(emc, -230.6640782865, 7)

    def test_mc1step_4o4e_df(self):
        mc = mcscf.CASSCF(mf, 4, 4).density_fit(auxbasis='weigend')
        mc.conv_tol = 1e-8
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -230.6617475545396, 7)

        mf1 = mf.density_fit(auxbasis='weigend')
        mc = mcscf.CASSCF(mf1, 4, 4).density_fit(auxbasis='weigend')
        mc.conv_tol = 1e-8
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -230.6617475545396, 7)

    def test_mc1step_9o8e(self):
        mc = mcscf.approx_hessian(mcscf.CASSCF(mf, 9, 8), auxbasis='weigend')
        mc.conv_tol = 1e-8
        mo = mf.mo_coeff.copy()
        mo[:,[15,16,17,18]] = mf.mo_coeff[:,[17,18,15,16]]
        emc = mc.mc1step(mo)[0]
        self.assertAlmostEqual(emc, -230.72211519779304, 6)
        #?self.assertAlmostEqual(emc, -230.72681659920534, 6)


if __name__ == "__main__":
    print("Full Tests for density fitting C6H6")
    unittest.main()
