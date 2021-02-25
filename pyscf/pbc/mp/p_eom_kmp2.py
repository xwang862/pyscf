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

import time
from functools import reduce
import sys
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.cc import eom_kccsd_rhf as eom_krccsd
from pyscf.pbc.cc import eom_kccsd_ghf as eom_kgccsd

from pyscf.pbc.cc.kintermediates_rhf import _new
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)  # noqa

einsum = lib.einsum

def eomee_mp2_singlet(eom, nroots=1, koopmans=False, guess=None, left=False,
                      eris=None, imds=None, diag=None,
                      kptlist=None, dtype=None):
    """See `eom_kgccsd.kernel()` for a description of arguments. 
    """
    eom.converged, eom.e, eom.v  \
            = eom_kgccsd.kernel_ee(eom, nroots, koopmans, guess, left, eris=eris,
                                   imds=imds, diag=diag, partition=eom.partition,
                                   kptlist=kptlist, dtype=dtype)
    return eom.e, eom.v


def _mem_usage(nkpts, nocc, nvir):
    incore = nkpts ** 3 * nocc * nvir ** 3
    # Roughly, factor of two for intermediates and factor of two
    # for safety (temp arrays, copying, etc)
    incore *= 4
    # TODO: Improve incore estimate and add outcore estimate
    outcore = basic = incore
    return incore * 16 / 1e6, outcore * 16 / 1e6, basic * 16 / 1e6


class PEOMMP2EESinglet(eom_krccsd.EOMEESinglet):
    """P–EOM–MBPT2 class where
    T1 = 0, 
    T2 = 1st-order t2 amplitudes, and
    doubles-doubles block of Hbar = \epsilon_a + \epsilon_b - \epsilon_i - \epsilon_j (0th-order effective Hamiltonian).

    Ref: Joshua J. Goings, Marco Caricato, Michael J. Frisch, and Xiaosong Li, J. Chem. Phys. 141, 164116 (2014)
    """
    def __init__(self, mp):
        self._mp = mp
        eom_krccsd.EOMEESinglet.__init__(self, mp)
        self.partition = 'mp'

    kernel = eomee_mp2_singlet

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self._mp, mo_coeff)

    def make_imds(self, eris=None, mo_coeff=None):
        if eris is None:
            eris = self.ao2mo(mo_coeff)
        imds = _IMDS(self._mp, eris)
        imds.make_ee()
        return imds


#######################################
#
# _ERIS.
#
# Note the two electron integrals (ab|cd) are stored as [ka,kc,kb,a,c,b,d] here
#
# Copied from kccsd_rhf and slightly modified
# TODO merge with kccsd_rhf
class _ERIS:
    def __init__(self, cc, mo_coeff=None, method='incore'):
        from pyscf.pbc import df
        from pyscf.pbc import tools
        from pyscf.pbc.cc.ccsd import _adjust_occ
        log = logger.Logger(cc.stdout, cc.verbose)
        cput0 = (time.clock(), time.time())
        cell = cc._scf.cell
        kpts = cc.kpts
        nkpts = cc.nkpts
        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc

        # if any(nocc != np.count_nonzero(cc._scf.mo_occ[k]>0)
        #       for k in range(nkpts)):
        #    raise NotImplementedError('Different occupancies found for different k-points')

        if mo_coeff is None:
            mo_coeff = cc.mo_coeff
        dtype = mo_coeff[0].dtype

        mo_coeff = self.mo_coeff = padded_mo_coeff(cc, mo_coeff)

        # Re-make our fock MO matrix elements from density and fock AO
        dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)

        keep_exxdiv = getattr(cc, "keep_exxdiv", False)            

        exxdiv = cc._scf.exxdiv if keep_exxdiv else None
        with lib.temporary_env(cc._scf, exxdiv=exxdiv):
            # _scf.exxdiv affects eris.fock. HF exchange correction should be
            # excluded from the Fock matrix.
            vhf = cc._scf.get_veff(cell, dm)
        fockao = cc._scf.get_hcore() + vhf
        self.fock = np.asarray([reduce(np.dot, (mo.T.conj(), fockao[k], mo))
                                for k, mo in enumerate(mo_coeff)])
        self.e_hf = cc._scf.energy_tot(dm=dm, vhf=vhf)

        self.mo_energy = [self.fock[k].diagonal().real for k in range(nkpts)]

        if not keep_exxdiv:
            self.mo_energy = [self.fock[k].diagonal().real for k in range(nkpts)]
            # Add HFX correction in the self.mo_energy to improve convergence in
            # CCSD iteration. It is useful for the 2D systems since their occupied and
            # the virtual orbital energies may overlap which may lead to numerical
            # issue in the CCSD iterations.
            # FIXME: Whether to add this correction for other exxdiv treatments?
            # Without the correction, MP2 energy may be largely off the correct value.
            madelung = tools.madelung(cell, kpts)
            self.mo_energy = [_adjust_occ(mo_e, nocc, -madelung)
                              for k, mo_e in enumerate(self.mo_energy)]

        # Get location of padded elements in occupied and virtual space.
        nocc_per_kpt = get_nocc(cc, per_kpoint=True)
        nonzero_padding = padding_k_idx(cc, kind="joint")

        # Check direct and indirect gaps for possible issues with CCSD convergence.
        mo_e = [self.mo_energy[kp][nonzero_padding[kp]] for kp in range(nkpts)]
        mo_e = np.sort([y for x in mo_e for y in x])  # Sort de-nested array
        gap = mo_e[np.sum(nocc_per_kpt)] - mo_e[np.sum(nocc_per_kpt)-1]
        if gap < 1e-5:
            logger.warn(cc, 'HOMO-LUMO gap %s too small for KCCSD. '
                            'May cause issues in convergence.', gap)

        mem_incore, mem_outcore, mem_basic = _mem_usage(nkpts, nocc, nvir)
        mem_now = lib.current_memory()[0]
        fao2mo = cc._scf.with_df.ao2mo

        kconserv = cc.khelper.kconserv
        khelper = cc.khelper
        orbo = np.asarray(mo_coeff[:,:,:nocc], order='C')
        orbv = np.asarray(mo_coeff[:,:,nocc:], order='C')

        if (method == 'incore' and (mem_incore + mem_now < cc.max_memory)
                or cell.incore_anyway):
            log.info('using incore ERI storage')
            # self.oooo = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=dtype)
            self.ooov = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=dtype)
            self.oovv = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=dtype)
            self.ovov = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=dtype)
            self.voov = np.empty((nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=dtype)
            self.vovv = np.empty((nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir), dtype=dtype)
            #self.vvvv = np.empty((nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), dtype=dtype)
            # self.vvvv = cc._scf.with_df.ao2mo_7d(orbv, factor=1./nkpts).transpose(0,2,1,3,5,4,6)

            for (ikp,ikq,ikr) in khelper.symm_map.keys():
                iks = kconserv[ikp,ikq,ikr]
                eri_kpt = fao2mo((mo_coeff[ikp],mo_coeff[ikq],mo_coeff[ikr],mo_coeff[iks]),
                                 (kpts[ikp],kpts[ikq],kpts[ikr],kpts[iks]), compact=False)
                if dtype == np.float: eri_kpt = eri_kpt.real
                eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)
                for (kp, kq, kr) in khelper.symm_map[(ikp, ikq, ikr)]:
                    eri_kpt_symm = khelper.transform_symm(eri_kpt, kp, kq, kr).transpose(0, 2, 1, 3)
                    # self.oooo[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, :nocc, :nocc] / nkpts
                    self.ooov[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, :nocc, nocc:] / nkpts
                    self.oovv[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, nocc:, nocc:] / nkpts
                    self.ovov[kp, kr, kq] = eri_kpt_symm[:nocc, nocc:, :nocc, nocc:] / nkpts
                    self.voov[kp, kr, kq] = eri_kpt_symm[nocc:, :nocc, :nocc, nocc:] / nkpts
                    self.vovv[kp, kr, kq] = eri_kpt_symm[nocc:, :nocc, nocc:, nocc:] / nkpts
                    #self.vvvv[kp, kr, kq] = eri_kpt_symm[nocc:, nocc:, nocc:, nocc:] / nkpts

            self.dtype = dtype
        else:
            log.info('using HDF5 ERI storage')
            self.feri1 = lib.H5TmpFile()

            # self.oooo = self.feri1.create_dataset('oooo', (nkpts, nkpts, nkpts, nocc, nocc, nocc, nocc), dtype.char)
            self.ooov = self.feri1.create_dataset('ooov', (nkpts, nkpts, nkpts, nocc, nocc, nocc, nvir), dtype.char)
            self.oovv = self.feri1.create_dataset('oovv', (nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype.char)
            self.ovov = self.feri1.create_dataset('ovov', (nkpts, nkpts, nkpts, nocc, nvir, nocc, nvir), dtype.char)
            self.voov = self.feri1.create_dataset('voov', (nkpts, nkpts, nkpts, nvir, nocc, nocc, nvir), dtype.char)
            self.vovv = self.feri1.create_dataset('vovv', (nkpts, nkpts, nkpts, nvir, nocc, nvir, nvir), dtype.char)

            # vvvv_required = ((not cc.direct)
            #                  # cc._scf.with_df needs to be df.GDF only (not MDF)
            #                  or type(cc._scf.with_df) is not df.GDF
            #                  # direct-vvvv for pbc-2D is not supported so far
            #                  or cell.dimension == 2)
            # if vvvv_required:
            #     self.vvvv = self.feri1.create_dataset('vvvv', (nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), dtype.char)
            # else:
            #     self.vvvv = None

            # <ij|pq>  = (ip|jq)
            cput1 = time.clock(), time.time()
            for kp in range(nkpts):
                for kq in range(nkpts):
                    for kr in range(nkpts):
                        ks = kconserv[kp, kq, kr]
                        orbo_p = mo_coeff[kp][:, :nocc]
                        orbo_r = mo_coeff[kr][:, :nocc]
                        buf_kpt = fao2mo((orbo_p, mo_coeff[kq], orbo_r, mo_coeff[ks]),
                                         (kpts[kp], kpts[kq], kpts[kr], kpts[ks]), compact=False)
                        if mo_coeff[0].dtype == np.float: buf_kpt = buf_kpt.real
                        buf_kpt = buf_kpt.reshape(nocc, nmo, nocc, nmo).transpose(0, 2, 1, 3)
                        self.dtype = buf_kpt.dtype
                        # self.oooo[kp, kr, kq, :, :, :, :] = buf_kpt[:, :, :nocc, :nocc] / nkpts
                        self.ooov[kp, kr, kq, :, :, :, :] = buf_kpt[:, :, :nocc, nocc:] / nkpts
                        self.oovv[kp, kr, kq, :, :, :, :] = buf_kpt[:, :, nocc:, nocc:] / nkpts
            cput1 = log.timer_debug1('transforming oopq', *cput1)

            # <ia|pq> = (ip|aq)
            cput1 = time.clock(), time.time()
            for kp in range(nkpts):
                for kq in range(nkpts):
                    for kr in range(nkpts):
                        ks = kconserv[kp, kq, kr]
                        orbo_p = mo_coeff[kp][:, :nocc]
                        orbv_r = mo_coeff[kr][:, nocc:]
                        buf_kpt = fao2mo((orbo_p, mo_coeff[kq], orbv_r, mo_coeff[ks]),
                                         (kpts[kp], kpts[kq], kpts[kr], kpts[ks]), compact=False)
                        if mo_coeff[0].dtype == np.float: buf_kpt = buf_kpt.real
                        buf_kpt = buf_kpt.reshape(nocc, nmo, nvir, nmo).transpose(0, 2, 1, 3)
                        self.ovov[kp, kr, kq, :, :, :, :] = buf_kpt[:, :, :nocc, nocc:] / nkpts
                        # TODO: compute vovv on the fly
                        self.vovv[kr, kp, ks, :, :, :, :] = buf_kpt[:, :, nocc:, nocc:].transpose(1, 0, 3, 2) / nkpts
                        self.voov[kr, kp, ks, :, :, :, :] = buf_kpt[:, :, nocc:, :nocc].transpose(1, 0, 3, 2) / nkpts
            cput1 = log.timer_debug1('transforming ovpq', *cput1)

            ## Without k-point symmetry
            # cput1 = time.clock(), time.time()
            # for kp in range(nkpts):
            #    for kq in range(nkpts):
            #        for kr in range(nkpts):
            #            ks = kconserv[kp,kq,kr]
            #            orbv_p = mo_coeff[kp][:,nocc:]
            #            orbv_q = mo_coeff[kq][:,nocc:]
            #            orbv_r = mo_coeff[kr][:,nocc:]
            #            orbv_s = mo_coeff[ks][:,nocc:]
            #            for a in range(nvir):
            #                orbva_p = orbv_p[:,a].reshape(-1,1)
            #                buf_kpt = fao2mo((orbva_p,orbv_q,orbv_r,orbv_s),
            #                                 (kpts[kp],kpts[kq],kpts[kr],kpts[ks]), compact=False)
            #                if mo_coeff[0].dtype == np.float: buf_kpt = buf_kpt.real
            #                buf_kpt = buf_kpt.reshape((1,nvir,nvir,nvir)).transpose(0,2,1,3)
            #                self.vvvv[kp,kr,kq,a,:,:,:] = buf_kpt[:] / nkpts
            # cput1 = log.timer_debug1('transforming vvvv', *cput1)

            # cput1 = time.clock(), time.time()
            # mem_now = lib.current_memory()[0]
            # if not vvvv_required:
            #     _init_df_eris(cc, self)

            # elif nvir ** 4 * 16 / 1e6 + mem_now < cc.max_memory:
            #     for (ikp, ikq, ikr) in khelper.symm_map.keys():
            #         iks = kconserv[ikp, ikq, ikr]
            #         orbv_p = mo_coeff[ikp][:, nocc:]
            #         orbv_q = mo_coeff[ikq][:, nocc:]
            #         orbv_r = mo_coeff[ikr][:, nocc:]
            #         orbv_s = mo_coeff[iks][:, nocc:]
            #         # unit cell is small enough to handle vvvv in-core
            #         buf_kpt = fao2mo((orbv_p,orbv_q,orbv_r,orbv_s),
            #                          kpts[[ikp,ikq,ikr,iks]], compact=False)
            #         if dtype == np.float: buf_kpt = buf_kpt.real
            #         buf_kpt = buf_kpt.reshape((nvir, nvir, nvir, nvir))
            #         for (kp, kq, kr) in khelper.symm_map[(ikp, ikq, ikr)]:
            #             buf_kpt_symm = khelper.transform_symm(buf_kpt, kp, kq, kr).transpose(0, 2, 1, 3)
            #             self.vvvv[kp, kr, kq] = buf_kpt_symm / nkpts
            # else:
            #     raise MemoryError('Minimal memory requirements %s MB'
            #                       % (mem_now + nvir ** 4 / 1e6 * 16 * 2))
            #     for (ikp, ikq, ikr) in khelper.symm_map.keys():
            #         for a in range(nvir):
            #             orbva_p = orbv_p[:, a].reshape(-1, 1)
            #             buf_kpt = fao2mo((orbva_p, orbv_q, orbv_r, orbv_s),
            #                              (kpts[ikp], kpts[ikq], kpts[ikr], kpts[iks]), compact=False)
            #             if mo_coeff[0].dtype == np.float: buf_kpt = buf_kpt.real
            #             buf_kpt = buf_kpt.reshape((1, nvir, nvir, nvir)).transpose(0, 2, 1, 3)

            #             self.vvvv[ikp, ikr, ikq, a, :, :, :] = buf_kpt[0, :, :, :] / nkpts
            #             # Store symmetric permutations
            #             self.vvvv[ikr, ikp, iks, :, a, :, :] = buf_kpt.transpose(1, 0, 3, 2)[:, 0, :, :] / nkpts
            #             self.vvvv[ikq, iks, ikp, :, :, a, :] = buf_kpt.transpose(2, 3, 0, 1).conj()[:, :, 0, :] / nkpts
            #             self.vvvv[iks, ikq, ikr, :, :, :, a] = buf_kpt.transpose(3, 2, 1, 0).conj()[:, :, :, 0] / nkpts
            # cput1 = log.timer_debug1('transforming vvvv', *cput1)

        log.timer('P-EOM-MP2 integral transformation', *cput0)


class _IMDS(eom_krccsd._IMDS):
    def __init__(self, mp, eris=None):
        nkpts, nocc, nvir = (mp.t2.shape[x] for x in (0, 3, 5))
        dtype = mp.t2.dtype
        mp.t1 = np.zeros((nkpts, nocc, nvir), dtype=dtype)
        eom_krccsd._IMDS.__init__(self, mp, eris)

    def _make_shared_1e(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t2, eris = self.t2, self.eris
        kconserv = self.kconserv
        self.Foo = Foo(t2, eris, kconserv)
        self.Fvv = Fvv(t2, eris, kconserv)
        self.Fov = Fov(t2, eris, kconserv)

        log.timer('EOM-MP2 shared one-electron intermediates', *cput0)

    def _make_shared_2e(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t2, eris = self.t2, self.eris
        kconserv = self.kconserv

        if self._fimd is not None:
            nkpts, nocc, nvir = (t2.shape[x] for x in (0, 3, 5))
            ovov_dest = self._fimd.create_dataset('ovov', (nkpts, nkpts, nkpts, nocc, nvir, nocc, nvir), t2.dtype.char)
            ovvo_dest = self._fimd.create_dataset('ovvo', (nkpts, nkpts, nkpts, nocc, nvir, nvir, nocc), t2.dtype.char)
        else:
            ovov_dest = ovvo_dest = None

        # 2 virtuals
        self.woVoV = Wovov(t2, eris, kconserv, ovov_dest)
        self.woVvO = Wovvo(t2, eris, kconserv, ovvo_dest)

        log.timer('EOM-MP2 shared two-electron intermediates', *cput0)

    def make_ip(self):
        raise NotImplementedError

    def make_ea(self):
        raise NotImplementedError

    def make_ee(self):
        self._make_shared_1e()
        if self._made_shared_2e is False:
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t2, eris = self.t2, self.eris
        kconserv = self.kconserv

        if self._fimd is not None:
            nkpts, nocc, nvir = t2.shape
            ovoo_dest = self._fimd.create_dataset('ovoo', (nkpts, nkpts, nkpts, nocc, nvir, nocc, nocc), t2.dtype.char)
            vvvo_dest = self._fimd.create_dataset('vvvo', (nkpts, nkpts, nkpts, nvir, nvir, nvir, nocc), t2.dtype.char)

        else:
            ovoo_dest = vvvo_dest = None

        # 0 or 1 virtuals, needed for IP, EE
        self.woOoV = eris.ooov
        self.woVoO = Wovoo(t2, eris, kconserv, ovoo_dest)
        # 3 or 4 virtuals, needed for EA, EE
        self.wvOvV = eris.vovv
        self.wvVvO = Wvvvo(t2, eris, kconserv, vvvo_dest)
        
        log.timer('EOM-MP2 EE intermediates', *cput0)



#######################################
#
# The following set of intermediate equations are similar to those in kintermediates_rhf
# 
# The only difference is that all t1-related parts are removed
#
def Foo(t2, eris, kconserv):
    nkpts, nocc, nvir = (t2.shape[x] for x in (0, 3, 5))
    Fki = np.empty((nkpts,nocc,nocc),dtype=t2.dtype)
    for ki in range(nkpts):
        kk = ki
        Fki[ki] = eris.fock[ki,:nocc,:nocc].copy()
        for kl in range(nkpts):
            for kc in range(nkpts):
                kd = kconserv[kk,kc,kl]
                Soovv = 2*eris.oovv[kk,kl,kc] - eris.oovv[kk,kl,kd].transpose(0,1,3,2)
                Fki[ki] += einsum('klcd,ilcd->ki',Soovv,t2[ki,kl,kc])
    return Fki

def Fvv(t2, eris, kconserv):
    nkpts, nocc, nvir = (t2.shape[x] for x in (0, 3, 5))
    Fac = np.empty((nkpts,nvir,nvir),dtype=t2.dtype)
    for ka in range(nkpts):
        kc = ka
        Fac[ka] = eris.fock[ka,nocc:,nocc:].copy()
        for kl in range(nkpts):
            for kk in range(nkpts):
                kd = kconserv[kk,kc,kl]
                Soovv = 2*eris.oovv[kk,kl,kc] - eris.oovv[kk,kl,kd].transpose(0,1,3,2)
                Fac[ka] += -einsum('klcd,klad->ac',Soovv,t2[kk,kl,ka])
    return Fac

def Fov(t2, eris, kconserv):
    nkpts, nocc, nvir = (t2.shape[x] for x in (0, 3, 5))
    Fkc = np.empty((nkpts,nocc,nvir),dtype=t2.dtype)
    Fkc[:] = eris.fock[:,:nocc,nocc:].copy()
    return Fkc

def Wovov(t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = (t2.shape[x] for x in (0, 3, 5))
    Wkbid = _new(eris.ovov.shape, t2.dtype, out)
    for kk in range(nkpts):
        for kb in range(nkpts):
            for ki in range(nkpts):
                kd = kconserv[kk,ki,kb]
                #   kk + kl - kc - kd = 0
                # => kc = kk - kd + kl
                ovov = eris.ovov[kk,kb,ki].copy()
                for kl in range(nkpts):
                    kc = kconserv[kk,kd,kl]
                    ovov -= einsum('klcd,ilcb->kbid',eris.oovv[kk,kl,kc],t2[ki,kl,kc])
                Wkbid[kk,kb,ki] = ovov
    return Wkbid

def Wovvo(t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = (t2.shape[x] for x in (0, 3, 5))
    Wkaci = _new((nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), t2.dtype, out)
    for kk in range(nkpts):
        for ka in range(nkpts):
            for kc in range(nkpts):
                ki = kconserv[kk,kc,ka]
                # ovvo[kk,ka,kc,ki] => voov[ka,kk,ki,kc]
                ovvo = np.asarray(eris.voov[ka,kk,ki]).transpose(1,0,3,2).copy()
                for kl in range(nkpts):
                    kd = kconserv[ki,ka,kl]
                    St2 = 2.*t2[ki,kl,ka] - t2[kl,ki,ka].transpose(1,0,2,3)
                    ovvo +=  einsum('klcd,ilad->kaci',eris.oovv[kk,kl,kc],St2)
                    ovvo += -einsum('kldc,ilad->kaci',eris.oovv[kk,kl,kd],t2[ki,kl,ka])
                Wkaci[kk,ka,kc] = ovvo
    return Wkaci

def Wovoo(t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = (t2.shape[x] for x in (0, 3, 5))
    FFov = eris.fock[:,:nocc,nocc:]

    Wkbij = _new((nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc), t2.dtype, out)
    for kk in range(nkpts):
        for kb in range(nkpts):
            for ki in range(nkpts):
                kj = kconserv[kk,ki,kb]
                ovoo = np.array(eris.ooov[ki,kj,kk]).transpose(2,3,0,1).conj()
                for kd in range(nkpts):
                    # kk + kl - ki - kd = 0
                    # => kl = ki - kk + kd
                    kl = kconserv[ki,kk,kd]
                    St2 = 2.*t2[kl,kj,kd] - t2[kj,kl,kd].transpose(1,0,2,3)
                    ovoo += einsum('klid,ljdb->kbij',  eris.ooov[kk,kl,ki], St2)
                    ovoo += einsum('lkid,ljdb->kbij', -eris.ooov[kl,kk,ki], t2[kl,kj,kd])
                    kl = kconserv[kb,ki,kd]
                    ovoo += einsum('lkjd,libd->kbij', -eris.ooov[kl,kk,kj], t2[kl,ki,kb])

                    # kb + kk - kd = kc
                    #kc = kconserv[kb,kd,kk]
                    ovoo += einsum('bkdc,jidc->kbij', eris.vovv[kb,kk,kd], t2[kj,ki,kd])

                ovoo += einsum('kc,ijcb->kbij',FFov[kk], t2[ki,kj,kk])
                Wkbij[kk,kb,ki] = ovoo
    return Wkbij

def Wvvvo(t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = (t2.shape[x] for x in (0, 3, 5))
    Wabcj = _new((nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc), t2.dtype, out)
    FFov = eris.fock[:,:nocc,nocc:]

    for ka in range(nkpts):
        for kb in range(nkpts):
            for kc in range(nkpts):
                kj = kconserv[ka,kc,kb]
                # vvvo[ka,kb,kc,kj] <= vovv[kc,kj,ka,kb].transpose(2,3,0,1).conj()
                vvvo = np.asarray(eris.vovv[kc,kj,ka]).transpose(2,3,0,1).conj()

                for kl in range(nkpts):
                    # ka + kl - kc - kd = 0
                    # => kd = ka - kc + kl
                    kd = kconserv[ka,kc,kl]
                    St2 = 2.*t2[kl,kj,kd] - t2[kl,kj,kb].transpose(0,1,3,2)
                    vvvo += einsum('alcd,ljdb->abcj',eris.vovv[ka,kl,kc], St2)
                    vvvo += einsum('aldc,ljdb->abcj',eris.vovv[ka,kl,kd], -t2[kl,kj,kd])
                    # kb - kc + kl = kd
                    kd = kconserv[kb,kc,kl]
                    vvvo += einsum('bldc,jlda->abcj',eris.vovv[kb,kl,kd], -t2[kj,kl,kd])

                    # kl + kk - kb - ka = 0
                    # => kk = kb + ka - kl
                    kk = kconserv[kb,kl,ka]
                    vvvo += einsum('lkjc,lkba->abcj',eris.ooov[kl,kk,kj],t2[kl,kk,kb])
                vvvo += einsum('lc,ljab->abcj',-FFov[kc],t2[kc,kj,ka])
                Wabcj[ka,kb,kc] = vvvo
    return Wabcj


if __name__ == '__main__':
    from pyscf.pbc import gto, scf, mp
    from pyscf.pbc.tools import lattice, pyscf_ase
    
    # Build unit cell
    cell = gto.Cell()
    ase_atom = lattice.get_ase_atom('c')
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell
    cell.unit = 'B'
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.verbose = 5
    cell.build()    

    # Run HF and MP2 with 1x1x2 Monkhorst-Pack k-point mesh
    kpts = cell.make_kpts([1,1,2])
    mymf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
    ehf = mymf.kernel()

    mymp = mp.KMP2(mymf)
    emp2 = mymp.kernel(with_t2=True)[0]

    # Run P-EOM-MP2
    myeom = PEOMMP2EESinglet(mymp)
    eeom = myeom.kernel(nroots=3, kptlist=[0])[0]
    print(eeom[0][0] - 0.28776335)
    print(eeom[0][1] - 0.29057406)
    print(eeom[0][2] - 0.29057978)