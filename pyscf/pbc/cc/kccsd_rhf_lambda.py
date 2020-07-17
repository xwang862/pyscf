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
Lambda equations for spin-restricted KCCSD.
'''
import time
import numpy
from pyscf import lib
from pyscf.cc import ccsd_lambda
from pyscf.lib import logger
from pyscf.pbc.mp.kmp2 import padding_k_idx
from pyscf.lib.parameters import LARGE_DENOM
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.cc.kccsd_rhf import _get_epq

einsum = lib.einsum

def kernel(cc, eris=None, t1=None, t2=None, l1=None, l2=None, max_cycle=50, tol=1e-8, verbose=logger.INFO):
    log = logger.Logger(cc.stdout, cc.verbose)
    log.info("******** PBC RCCSD lambda solver ********")
    return ccsd_lambda.kernel(cc, eris, t1, t2, l1, l2, max_cycle=max_cycle, tol=tol, verbose=verbose, fintermediates=make_intermediates, fupdate=update_lambda)

def make_intermediates(cc, t1=None, t2=None, eris=None):
    from pyscf.pbc.cc.eom_kccsd_rhf import _IMDS
    if eris is None: eris = cc.ao2mo(cc.mo_coeff)
    imds = _IMDS(cc, eris)
    imds.make_ee()
    return imds

def update_lambda(cc, t1, t2, l1, l2, eris, imds):
    """
    Spin adapted version of the following equations:
        Gauss and Stanton, J. Chem. Phys. 103, 3561 (1995) Table II. Unperturbed lambda equations (a) and (b)
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

    Gvv = numpy.zeros_like(Ftmp_vv)
    for ka, km, kn in kpts_helper.loop_kkk(nkpts):
        # G_ae <- - 2 l_mnaf t_mnef
        #  ka - ke = 0
        ke = ka
        Gvv[ka] -= 2. * einsum('mnaf,mnef->ae', l2[km, kn, ka], t2[km, kn, ke])
        # G_ae <- l_mnaf t_nmef
        Gvv[ka] += einsum('mnaf,nmef->ae', l2[km, kn, ka], t2[kn, km, ke]) 
    
    Goo = numpy.zeros_like(Ftmp_oo)
    for km, kn, ke in kpts_helper.loop_kkk(nkpts):
        # G_mi <= 2 l_inef t_mnef
        #  km - ki = 0
        ki = km
        Goo[km] += 2. * einsum('inef,mnef->mi', l2[ki, kn, ke], t2[km, kn, ke])
        # G_mi <= - l_inef t_mnfe
        #  km + kn - kf - ke = 0
        kf = kconserv[km, ke, kn]
        Goo[km] -= einsum('inef,mnfe->mi', l2[ki, kn, ke], t2[km, kn, kf])

    # l_ia <- F_ia
    l1new = numpy.copy(imds.Fov)

    for ki in range(nkpts):
        # ki - ka = 0
        ka = ki
        # l_ia <- - l_ma Ftmp_im
        # km = ka
        l1new[ki] -= einsum('ma,im->ia', l1[ka], Ftmp_oo[ki])
        # l_ia <- l_ie Ftmp_ea
        #  ke = ka
        l1new[ki] += einsum('ie,ea->ia', l1[ki], Ftmp_vv[ka])

        for ke in range(nkpts):
            # l_ia <- 2 l_me W_ieam
            #  km - ke = 0
            km = ke
            l1new[ki] += 2. * einsum('me,ieam->ia', l1[km], imds.woVvO[ki, ke, ka])
            # l_ia <- - l_me W_iema
            l1new[ki] -= einsum('me,iema->ia', l1[km], imds.woVoV[ki, ke, km])

            for kf in range(nkpts):
                # l_ia <- 2 l_imef W_efam
                #  ke + kf - ka - km = 0
                km = kconserv[ke, ka, kf]
                l1new[ki] += 2. * einsum('imef,efam->ia', l2[ki, km, ke], imds.wvVvO[ke, kf, ka])
                # l_ia <- - l_imef W_feam
                l1new[ki] -= einsum('imef,feam->ia', l2[ki, km, ke], imds.wvVvO[kf, ke, ka])

            for km in range(nkpts):
                # l_ia <- -2 l_mnae W_iemn
                #  ki + ke - km - kn = 0
                kn = kconserv[ki, km, ke]
                l1new[ki] -= 2. * einsum('mnae,iemn->ia', l2[km, kn, ka], imds.woVoO[ki, ke, km])
                # l_ia <- l_mnae W_ienm
                l1new[ki] += einsum('mnae,ienm->ia', l2[km, kn, ka], imds.woVoO[ki, ke, kn])

        for ke in range(nkpts):
            # l_ia <- - 2 G_ef W_eifa
            #  ke - kf = 0
            kf = ke
            l1new[ki] -= 2. * einsum('ef,eifa->ia', Gvv[ke], imds.wvOvV[ke, ki, kf])
            # l_ia <- G_ef W_iefa = G_ef W_eiaf
            l1new[ki] += einsum('ef,eiaf->ia', Gvv[ke], imds.wvOvV[ke, ki, ka])

        for km in range(nkpts):
            # l_ia <- - 2 G_mn W_mina
            #  km - kn = 0
            kn = km
            l1new[ki] -= 2. * einsum('mn,mina->ia', Goo[km], imds.woOoV[km, ki, kn])
            # l_ia <- G_mn W_imna
            l1new[ki] += einsum('mn,imna->ia', Goo[km], imds.woOoV[ki, km, kn])

    # l_ijab <- V_ijab
    l2new = numpy.copy(eris.oovv)

    # Divide L1 by epsilon_ia
    for ki in range(nkpts):
        # ki - ka = 0
        ka = ki
        # Remove zero/padded elements from denominator
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        l1new[ki] /= eia

    # Divide L2 by epsilon_ijab
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])

        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                       [0,nvir,kb,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        eijab = eia[:, None, :, None] + ejb[:, None, :]

        l2new[ki, kj, ka] /= eijab

    time0 = log.timer_debug1('update l1 l2', *time0)

    return l1new, l2new
    