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
EOM-EE-CCSD (singlet) correction-vector spectra of eom_kccsd_rhf.EOMEESinglet.get_spectrum (eeccsd_spectrum_singlet) on
k-meshes: the momentum operator (optical absorption, qvec=None) and the density operator exp(iQ.r) (finite momentum
transfer), and the wrappers get_absorption_spectrum, get_dynamic_structure_factor and get_energy_loss_function.
A k-mesh calculation equals the Gamma-point calculation of the equivalent supercell, where every Q of the mesh is
a reciprocal vector (zero momentum transfer): the two use different code paths (kshift != 0 vs kshift = 0 with the
elastic Bragg pole removed). The 3x1x1 mesh has no self-inverse kshift, so Q and -Q, and k + Q and k - Q, differ.
H2O in a small cubic box has no inversion center.
'''

import unittest
import numpy as np
from pyscf.pbc import gto, scf, cc
from pyscf.pbc.cc import eom_kccsd_rhf
from pyscf.pbc.tools.pbc import super_cell

HARTREE2EV = 27.211386245988


def setUpModule():
    global cell, w, eta, eom_k, eom_sc
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
    w = np.array([9., 11., 12.5, 14.5]) / HARTREE2EV
    eta = 0.1 / HARTREE2EV
    eom_k = make_eom(cell, cell.make_kpts([3, 1, 1]))
    eom_sc = make_eom(super_cell(cell, [3, 1, 1]), np.zeros((1, 3)))

def tearDownModule():
    global cell, eom_k, eom_sc
    del cell, eom_k, eom_sc


def make_eom(cell, kpts):
    kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit()
    kmf.conv_tol = 1e-11
    kmf.kernel()
    mycc = cc.KRCCSD(kmf)
    mycc.conv_tol = 1e-11
    mycc.conv_tol_normt = 1e-9
    mycc.kernel()
    mycc.solve_lambda()
    return eom_kccsd_rhf.EOMEESinglet(mycc)

def spectrum(eom, qvec=None, eta=None, scan=None):
    if eta is None: eta = globals()['eta']
    if scan is None: scan = w
    return eom.get_spectrum(scan, eta, qvec=qvec, tol=1e-9, maxiter=500)


class KnownValues(unittest.TestCase):
    def test_density_kmesh_vs_supercell(self):
        kpts = eom_k.kpts
        b = cell.reciprocal_vectors()
        q1, q2 = -(kpts[1] - kpts[0]), -(kpts[2] - kpts[0])
        for qvec, kshift in ((q1, 1), (q2, 2), (q1 + b[0] + 2 * b[1], 1), (b[0], 0)):
            self.assertEqual(eom_kccsd_rhf.find_kshift(eom_k, qvec), kshift)
            self.assertEqual(eom_kccsd_rhf.find_kshift(eom_sc, qvec), 0)
            s_k = spectrum(eom_k, qvec)[0][0]
            s_sc = spectrum(eom_sc, qvec)[0][0]
            self.assertGreater(abs(s_sc).max(), 0.1)
            self.assertAlmostEqual(abs(s_k - s_sc).max() / abs(s_sc).max(), 0, 6)

    def test_optical_kmesh_vs_supercell(self):
        s_k = spectrum(eom_k)[0].sum(axis=0)
        s_sc = spectrum(eom_sc)[0].sum(axis=0)
        self.assertAlmostEqual(abs(s_k - s_sc).max() / abs(s_sc).max(), 0, 6)

    def test_density_time_reversal(self):
        # Time reversal maps rho_Q to rho_-Q and conjugates the CC equations, so the response functions satisfy
        # G_-Q(z) = conj(G_Q(z*)), i.e. S(-Q, w; eta) = -S(Q, w; -eta). (S(-Q) = S(Q) only holds for Hermitian
        # transition strengths; the EOM-CCSD ones, <0|(1+L) rho^+|R_n><L_n|rho|0>, are complex.)
        # Q and -Q are at different kshifts, with independently diagonalized orbitals at k and -k.
        kpts = eom_k.kpts
        q1 = -(kpts[1] - kpts[0])
        self.assertEqual(eom_kccsd_rhf.find_kshift(eom_k, -q1), 2)
        s_m = spectrum(eom_k, -q1)[0][0]
        s_p = spectrum(eom_k, q1, eta=-eta)[0][0]
        self.assertGreater(abs(s_m).max(), 0.1)
        self.assertAlmostEqual(abs(s_m + s_p).max() / abs(s_m).max(), 0, 6)

    def test_named_wrappers(self):
        # get_absorption_spectrum: the momentum spectrum as is (old positional call kept working);
        # get_dynamic_structure_factor: per electron, S / (N_k N_e); get_energy_loss_function: 4 pi^2 S / (|Q|^2 N_k V_cell)
        scan = w[1:2]
        s_p = spectrum(eom_k, scan=scan)[0]
        s_abs = eom_k.get_absorption_spectrum(scan, eta, None, 1e-9, 500)[0]
        self.assertAlmostEqual(abs(s_abs - s_p).max() / abs(s_p).max(), 0, 9)
        kpts = eom_k.kpts
        qvec = -(kpts[1] - kpts[0])
        s_q = spectrum(eom_k, qvec, scan=scan)[0][0]
        nk, nelec, vol = len(kpts), cell.nelectron, cell.vol
        dsf = eom_k.get_dynamic_structure_factor(scan, eta, qvec, tol=1e-9, maxiter=500)[0]
        loss = eom_k.get_energy_loss_function(scan, eta, qvec, tol=1e-9, maxiter=500)[0]
        self.assertEqual(dsf.shape, scan.shape)
        self.assertAlmostEqual(abs(dsf - s_q / (nk * nelec)).max() / abs(dsf).max(), 0, 9)
        ref = 4 * np.pi**2 * s_q / (np.dot(qvec, qvec) * nk * vol)
        self.assertAlmostEqual(abs(loss - ref).max() / abs(ref).max(), 0, 9)

    def test_deprecated_alias(self):
        scan = w[1:2]
        with self.assertWarns(DeprecationWarning):
            s_old = eom_kccsd_rhf.optical_absorption_singlet(eom_k, scan, eta, tol=1e-9, maxiter=500)[0]
        s_new = spectrum(eom_k, scan=scan)[0]
        self.assertAlmostEqual(abs(s_old - s_new).max() / abs(s_new).max(), 0, 9)

    def test_invalid_input(self):
        kpts = eom_k.kpts
        qvec = -(kpts[1] - kpts[0])
        with self.assertRaises(ValueError):   # not commensurate with the k-mesh
            eom_k.get_spectrum(w, eta, qvec=0.3 * (kpts[1] - kpts[0]))
        with self.assertRaises(ValueError):   # qvec at kshift 1, kshift given as 0
            eom_k.get_spectrum(w, eta, qvec=qvec, kshift=0)
        with self.assertRaises(ValueError):   # momentum operator only at kshift 0
            eom_k.get_spectrum(w, eta, kshift=1)
        with self.assertRaises(ValueError):   # absorption spectrum: momentum operator only
            eom_k.get_absorption_spectrum(w, eta, qvec=qvec)
        for f in (eom_k.get_dynamic_structure_factor, eom_k.get_energy_loss_function):
            with self.assertRaises(ValueError):   # a nonzero momentum transfer is required
                f(w, eta, None)
            with self.assertRaises(ValueError):
                f(w, eta, np.zeros(3))


if __name__ == '__main__':
    print('EOM-EE-CCSD spectra on k-meshes: optical and finite momentum transfer')
    unittest.main()
