#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
#
# Author: Xiao Wang <xiaowang314159@gmail.com>
#

'''
Lambda equations for spin orbital KCCSD.

Ref: Gauss and Stanton, J. Chem. Phys. 103, 3561 (1995)
'''

import time
import numpy
from pyscf import lib
from pyscf.cc import ccsd_lambda
from pyscf.lib import logger
from pyscf.pbc.mp.kmp2 import padding_k_idx
from pyscf.lib.parameters import LARGE_DENOM
from pyscf.pbc.lib import kpts_helper

einsum = lib.einsum

def kernel(cc, eris=None, t1=None, t2=None, l1=None, l2=None, max_cycle=50, tol=1e-8, verbose=logger.INFO):
    log = logger.Logger(cc.stdout, cc.verbose)
    log.info("******** PBC GCCSD lambda solver ********")
    return ccsd_lambda.kernel(cc, eris, t1, t2, l1, l2, max_cycle=max_cycle, tol=tol, verbose=verbose, fintermediates=make_intermediates, fupdate=update_lambda)


def make_intermediates(cc, t1=None, t2=None, eris=None):
    from pyscf.pbc.cc.eom_kccsd_ghf import _IMDS
    if eris is None: eris = cc.ao2mo(cc.mo_coeff)
    imds = _IMDS(cc, eris)
    imds.make_ee()
    return imds


def update_lambda(cc, t1, t2, l1, l2, eris, imds):
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    nkpts, nocc, nvir = t1.shape
    kconserv = imds.kconserv
    fock = eris.fock

    mo_e_o = [e[:nocc] for e in eris.mo_energy]
    mo_e_v = [e[nocc:] + cc.level_shift for e in eris.mo_energy]

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind="split")

    Ftmp_oo = imds.Foo - fock[:, :nocc, :nocc]
    Ftmp_vv = imds.Fvv - fock[:, nocc:, nocc:]

    # L1 equations
    
    # l1_ia <- F_ia
    l1new = numpy.copy(imds.Fov)

    for ki in range(nkpts):
        # ki - ka = 0
        ka = ki
        # l1_ia <- - l1_ma Ftmp_im
        #  km = ka 
        l1new[ki] -= einsum('ma,im->ia', l1[ka], Ftmp_oo[ki])
        # l1_ia <- l1_ie Ftmp_ea
        #  ke = ka
        l1new[ki] += einsum('ie,ea->ia', l1[ki], Ftmp_vv[ka])

    # L2 equations

    # l2_ijab <- V_ijab
    l2new = numpy.copy(eris.oovv)

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        # ki - ka + kj - kb = 0
        kb = kconserv[ki, ka, kj]
        # l2_ijab <- P(ab) l2_ijae Ftmp_eb
        #  ke = kb
        tmp = einsum('ijae,eb->ijab', l2[ki, kj, ka], Ftmp_vv[kb])
        l2new[ki, kj, ka] += tmp
        l2new[ki, kj, kb] -= tmp.transpose(0, 1, 3, 2)

        # l2_ijab <- - P(ij) l2_imab Ftmp_jm
        #  km = kj
        tmp = -1. * einsum('imab,jm->ijab', l2[ki, kj, ka], Ftmp_oo[kj])
        l2new[ki, kj, ka] += tmp
        l2new[kj, ki, ka] -= tmp.transpose(1, 0, 2, 3)


    # Divide L1 by epsilon_ia
    for ki in range(nkpts):
        # ki - ka = 0
        ka = ki
        # Remove zero/padded elements from denominator
        eia = LARGE_DENOM * numpy.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
        n0_ovp_ia = numpy.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
        eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]
        l1new[ki] /= eia   

    # Divide L2 by epsilon_ijab
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        # ki - ka + kj - kb = 0
        kb = kconserv[ki, ka, kj]
        eia = LARGE_DENOM * numpy.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
        n0_ovp_ia = numpy.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
        eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

        ejb = LARGE_DENOM * numpy.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
        n0_ovp_jb = numpy.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
        ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]
        eijab = eia[:, None, :, None] + ejb[:, None, :]

        l2new[ki, kj, ka] /= eijab             

    time0 = log.timer_debug1('update l1 l2', *time0)

    return l1new, l2new