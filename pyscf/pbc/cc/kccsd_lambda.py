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
    """
    Ref: Gauss and Stanton, J. Chem. Phys. 103, 3561 (1995) Table II. Unperturbed lambda equations (a) and (b)
    """
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    nkpts, nocc, nvir = t1.shape
    kconserv = imds.kconserv
    fock = eris.fock

    mo_e_o = [e[:nocc] for e in eris.mo_energy]
    mo_e_v = [e[nocc:] + cc.level_shift for e in eris.mo_energy]

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind="split")

    # Make some intermediates first
    # Ftmp_im = F_im - f_im
    Ftmp_oo = imds.Foo - fock[:, :nocc, :nocc]
    # Ftmp_ea = F_ea - f_ea
    Ftmp_vv = imds.Fvv - fock[:, nocc:, nocc:]

    # G_ae = -0.5 * l_mnaf t_mnef
    Gvv = numpy.zeros_like(Ftmp_vv)
    for ka, km, kn in kpts_helper.loop_kkk(nkpts):
        # ka - ke = 0
        ke = ka
        Gvv[ka] -= 0.5 * einsum('mnaf,mnef->ae', l2[km, kn, ka], t2[km, kn, ke])
    # G_mi = 0.5 * l_inef t_mnef
    Goo = numpy.zeros_like(Ftmp_oo)
    for km, kn, ke in kpts_helper.loop_kkk(nkpts):
        # km - ki = 0
        ki = km
        Goo[km] += 0.5 * einsum('inef,mnef->mi', l2[ki, kn, ke], t2[km, kn, ke])

    # L1 equations

    # l_ia <- F_ia
    l1new = numpy.copy(imds.Fov)

    for ki in range(nkpts):
        # ki - ka = 0
        ka = ki
        # l_ia <- - l_ma Ftmp_im
        #  km = ka 
        l1new[ki] -= einsum('ma,im->ia', l1[ka], Ftmp_oo[ki])
        # l_ia <- l_ie Ftmp_ea
        #  ke = ka
        l1new[ki] += einsum('ie,ea->ia', l1[ki], Ftmp_vv[ka])

        for ke in range(nkpts):
            # l_ia <- l_me W_ieam
            #  ki + ke - ka - km = 0
            km = kconserv[ki, ka, ke]
            l1new[ki] += einsum('me,ieam->ia', l1[km], imds.Wovvo[ki, ke, ka])

            for kf in range(nkpts):
                # l_ia <- 0.5 * l_imef W_efam
                #  ke + kf - ka - km = 0
                km = kconserv[ke, ka, kf]
                l1new[ki] += 0.5 * einsum('imef,efam->ia', l2[ki, km, ke], imds.Wvvvo[ke, kf, ka])

            for km in range(nkpts):
                # l_ia <- -0.5 * l_mnae W_iemn
                #  ki + ke - km - kn = 0
                kn = kconserv[ki, km, ke]
                l1new[ki] -= 0.5 * einsum('mnae,iemn->ia', l2[km, kn, ka], imds.Wovoo[ki, ke, km])

        for ke in range(nkpts):
            # l_ia <- - G_ef W_eifa
            #  ke - kf = 0
            kf = ke
            l1new[ki] -= einsum('ef,eifa->ia', Gvv[ke], imds.Wvovv[ke, ki, kf])

        for km in range(nkpts):
            # l_ia <- - G_mn W_mina
            #  km - kn = 0
            kn = km
            l1new[ki] -= einsum('mn,mina->ia', Goo[km], imds.Wooov[km, ki, kn])

    # L2 equations

    # l_ijab <- V_ijab
    l2new = numpy.copy(eris.oovv)

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        # ki - ka + kj - kb = 0
        kb = kconserv[ki, ka, kj]

        # l_ijab <- P(ab) l_ijae Ftmp_eb
        #  ke - kb = 0
        ke = kb
        tmp = einsum('ijae,eb->ijab', l2[ki, kj, ka], Ftmp_vv[ke])
        # l_ijab <- - P(ab) l_ma W_ijmb
        #  km - ka = 0
        km = ka
        tmp -= einsum('ma,ijmb->ijab', l1[km], imds.Wooov[ki, kj, km])
        # l_ijab <- P(ab) G_be V_ijae
        tmp += einsum('be,ijae->ijab', Gvv[kb], eris.oovv[ki, kj, ka])

        l2new[ki, kj, ka] += tmp
        l2new[ki, kj, kb] -= tmp.transpose(0, 1, 3, 2)

        # l_ijab <- - P(ij) l_imab Ftmp_jm
        #  kj - km = 0
        km = kj
        tmp = -1. * einsum('imab,jm->ijab', l2[ki, km, ka], Ftmp_oo[kj])
        # l_ijab <- P(ij) l_ie W_ejab
        #  ki - ke = 0
        ke = ki
        tmp += einsum('ie,ejab->ijab', l1[ki], imds.Wvovv[ke, kj, ka])
        # l_ijab <- - P(ij) G_mj V_imab
        #  km - kj = 0
        km = kj
        tmp -= einsum('mj,imab->ijab', Goo[km], eris.oovv[ki, km, ka])    

        l2new[ki, kj, ka] += tmp
        l2new[kj, ki, ka] -= tmp.transpose(1, 0, 2, 3)

        # l_ijab <- P(ij) P(ab) F_jb l_ia
        tmp = einsum('jb,ia->ijab', imds.Fov[kj], l1[ki])
        l2new[ki, kj, ka] += tmp
        l2new[kj, ki, ka] -= tmp.transpose(1, 0, 2, 3)
        l2new[ki, kj, kb] -= tmp.transpose(0, 1, 3, 2)
        l2new[kj, ki, kb] += tmp.transpose(1, 0, 3, 2)

        for km in range(nkpts):
            # l_ijab <- 0.5 l_mnab W_ijmn
            #  ki + kj - km - kn = 0
            kn = kconserv[ki, km, kj]
            l2new[ki, kj, ka] += 0.5 * einsum('mnab,ijmn->ijab', l2[km, kn, ka], imds.Woooo[ki, kj, km])

            ke = km
            # l_ijab <- 0.5 l_ijef W_efab
            #  ki + kj - ke - kf = 0
            kf = kconserv[ki, ke, kj]
            l2new[ki, kj, ka] += 0.5 * einsum('ijef,efab->ijab', l2[ki, kj, ke], imds.Wvvvv[ke, kf, ka])

            # l_ijab <- P(ij) P(ab) l_imae W_jebm
            #  ki + km - ka - ke = 0
            ke = kconserv[ki, ka, km]
            tmp = einsum('imae,jebm->ijab', l2[ki, km, ka], imds.Wovvo[kj, ke, kb])
            l2new[ki, kj, ka] += tmp
            l2new[kj, ki, ka] -= tmp.transpose(1, 0, 2, 3)
            l2new[ki, kj, kb] -= tmp.transpose(0, 1, 3, 2)
            l2new[kj, ki, kb] += tmp.transpose(1, 0, 3, 2)

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