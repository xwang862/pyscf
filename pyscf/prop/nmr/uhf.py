#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic NMR shielding tensor
'''

import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.scf import ucphf
from pyscf.ao2mo import _ao2mo
from pyscf.scf.newton_ah import _gen_uhf_response
from pyscf.prop.nmr import rhf as rhf_nmr


def dia(mol, dm0, gauge_orig=None, shielding_nuc=None):
    if not (isinstance(dm0, numpy.ndarray) and dm0.ndim == 2):
        dm0 = dm0[0] + dm0[1]
    return rhf_nmr.dia(mol, dm0, gauge_orig, shielding_nuc)

def para(mol, mo10, mo_coeff, mo_occ, shielding_nuc=None):
    if shielding_nuc is None:
        shielding_nuc = range(mol.natm)
    para_vir = numpy.empty((len(shielding_nuc),3,3))
    para_occ = numpy.empty((len(shielding_nuc),3,3))
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = mo_occ[0] == 0
    viridxb = mo_occ[1] == 0
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]
    nao = mo_coeff[0].shape[0]
    dm10_oo = numpy.empty((3,nao,nao))
    dm10_vo = numpy.empty((3,nao,nao))
    for i in range(3):
        dm10_oo[i] = reduce(numpy.dot, (orboa, mo10[0][i][occidxa], orboa.conj().T))
        dm10_oo[i]+= reduce(numpy.dot, (orbob, mo10[1][i][occidxb], orbob.conj().T))
        dm10_vo[i] = reduce(numpy.dot, (orbva, mo10[0][i][viridxa], orboa.conj().T))
        dm10_vo[i]+= reduce(numpy.dot, (orbvb, mo10[1][i][viridxb], orbob.conj().T))
    for n, atm_id in enumerate(shielding_nuc):
        mol.set_rinv_origin(mol.atom_coord(atm_id))
        h01 = mol.intor_asymmetric('int1e_prinvxp', 3)
        para_occ[n] = numpy.einsum('xji,yij->xy', dm10_oo, h01) * 2
        para_vir[n] = numpy.einsum('xji,yij->xy', dm10_vo, h01) * 2
    msc_para = para_occ + para_vir
    return msc_para, para_vir, para_occ

def make_h10(mol, dm0, gauge_orig=None, verbose=logger.WARN):
    log = logger.new_logger(mol, verbose=verbose)
    if gauge_orig is None:
        # A10_i dot p + p dot A10_i consistents with <p^2 g>
        # A10_j dot p + p dot A10_j consistents with <g p^2>
        # A10_j dot p + p dot A10_j => i/2 (rjxp - pxrj) = irjxp
        log.debug('First-order GIAO Fock matrix')
        h1 = -.5 * mol.intor('int1e_giao_irjxp', 3) + make_h10giao(mol, dm0)
    else:
        mol.set_common_origin(gauge_orig)
        h1 = -.5 * mol.intor('int1e_cg_irxp', 3)
        h1 = (h1, h1)
    return h1

def make_h10giao(mol, dm0):
    vj, vk = rhf_nmr.get_jk(mol, dm0)
    h1 = vj[0] + vj[1] - vk
    h1 -= mol.intor_asymmetric('int1e_ignuc', 3)
    h1 -= mol.intor('int1e_igkin', 3)
    return h1

def solve_mo1(mo_energy, mo_occ, h1, s1):
    '''uncoupled first order equation'''
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = ~occidxa
    viridxb = ~occidxb
    nocca = numpy.count_nonzero(occidxa)
    noccb = numpy.count_nonzero(occidxb)
    nmoa, nmob = mo_occ[0].size, mo_occ[1].size
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    eai_a = mo_energy[0][viridxa,None] - mo_energy[0][occidxa]
    eai_b = mo_energy[1][viridxb,None] - mo_energy[1][occidxb]
    s1_a = s1[0].reshape(-1,nmoa,nocca)
    s1_b = s1[1].reshape(-1,nmob,noccb)
    hs_a = mo1_a = h1[0].reshape(-1,nmoa,nocca) - s1_a * mo_energy[0][occidxa]
    hs_b = mo1_b = h1[1].reshape(-1,nmob,noccb) - s1_b * mo_energy[1][occidxb]
    mo_e1_a = hs_a[:,occidxa].copy()
    mo_e1_b = hs_b[:,occidxb].copy()

    mo1_a[:,viridxa]/= -eai_a
    mo1_b[:,viridxb]/= -eai_b
    mo1_a[:,occidxa] = -s1_a[:,occidxa] * .5
    mo1_b[:,occidxb] = -s1_b[:,occidxb] * .5
    mo_e1_a += mo1_a[:,occidxa] * (mo_energy[0][occidxa,None] - mo_energy[0][occidxa])
    mo_e1_b += mo1_b[:,occidxb] * (mo_energy[1][occidxb,None] - mo_energy[1][occidxb])
    return (mo1_a, mo1_b), (mo_e1_a, mo_e1_b)


class NMR(rhf_nmr.NMR):

    def shielding(self, mo1=None):
        if hasattr(self._scf, 'spin_square'):
            s2 = self._scf.spin_square()[0]
            if s2 > 1e-4:
                logger.warn(self, '<S^2> = %s. UHF-NMR shielding may have large error.\n'
                            'paramagnetic NMR should include this result plus '
                            'g-tensor and HFC tensors.', s2)
        return rhf_nmr.NMR.shielding(self, mo1)

    def dia(self, mol=None, dm0=None, gauge_orig=None, shielding_nuc=None):
        if mol is None: mol = self.mol
        if dm0 is None: dm0 = self._scf.make_rdm1()
        if not (isinstance(dm0, numpy.ndarray) and dm0.ndim == 2):
            # spin-traced 1pdm
            dm0 = dm0[0] + dm0[1]
        return rhf_nmr.dia(mol, dm0, gauge_orig, shielding_nuc)

    def para(self, mol=None, mo10=None, mo_coeff=None, mo_occ=None,
             shielding_nuc=None):
        if mol is None:           mol = self.mol
        if mo_coeff is None:      mo_coeff = self._scf.mo_coeff
        if mo_occ is None:        mo_occ = self._scf.mo_occ
        if shielding_nuc is None: shielding_nuc = self.shielding_nuc
        if mo10 is None:
            self.mo10, self.mo_e10 = self.solve_mo1()
            mo10 = self.mo10
        return para(mol, mo10, mo_coeff, mo_occ, shielding_nuc)

    def make_h10(self, mol=None, dm0=None, gauge_orig=None):
        if mol is None: mol = self.mol
        if dm0 is None: dm0 = self._scf.make_rdm1()
        if gauge_orig is None: gauge_orig = self.gauge_orig
        log = logger.Logger(self.stdout, self.verbose)
        h1 = make_h10(mol, dm0, gauge_orig, log)
        lib.chkfile.dump(self.chkfile, 'nmr/h1', h1)
        return h1

    def solve_mo1(self, mo_energy=None, mo_occ=None, h1=None, s1=None,
                  with_cphf=None):
        cput1 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        if mo_energy is None: mo_energy = self._scf.mo_energy
        if mo_occ    is None: mo_occ = self._scf.mo_occ
        if with_cphf is None: with_cphf = self.cphf

        mol = self.mol
        mo_coeff = self._scf.mo_coeff
        orboa = mo_coeff[0][:,mo_occ[0]>0]
        orbob = mo_coeff[1][:,mo_occ[1]>0]
        if h1 is None:
            dm0 = self._scf.make_rdm1(mo_coeff, mo_occ)
            h1 = self.make_h10(mol, dm0)
            h1a = [reduce(numpy.dot, (mo_coeff[0].T.conj(), x, orboa)) for x in h1[0]]
            h1b = [reduce(numpy.dot, (mo_coeff[1].T.conj(), x, orbob)) for x in h1[1]]
            h1 = (numpy.asarray(h1a), numpy.asarray(h1b))
        if s1 is None:
            s1 = self.make_s10(mol)
            s1a = [reduce(numpy.dot, (mo_coeff[0].T.conj(), x, orboa)) for x in s1]
            s1b = [reduce(numpy.dot, (mo_coeff[1].T.conj(), x, orbob)) for x in s1]
            s1 = (numpy.asarray(s1a), numpy.asarray(s1b))

        cput1 = log.timer('first order Fock matrix', *cput1)
        if self.cphf:
            vind = self.gen_vind(self._scf, mo_coeff, mo_occ)
            mo10, mo_e10 = ucphf.solve(vind, mo_energy, mo_occ, h1, s1,
                                       self.max_cycle_cphf, self.conv_tol,
                                       verbose=log)
        else:
            mo10, mo_e10 = solve_mo1(mo_energy, mo_occ, h1, s1)
        logger.timer(self, 'solving mo1 eqn', *cput1)
        return mo10, mo_e10

    def gen_vind(self, mf, mo_coeff, mo_occ):
        '''Induced potential'''
        vresp = _gen_uhf_response(self._scf, hermi=2)
        occidxa = mo_occ[0] > 0
        occidxb = mo_occ[1] > 0
        orboa = mo_coeff[0][:,occidxa]
        orbob = mo_coeff[1][:,occidxb]
        nocca = orboa.shape[1]
        noccb = orbob.shape[1]
        nao, nmo = mo_coeff[0].shape
        nvira = nmo - nocca
        def vind(mo1):
            mo1a = mo1.reshape(3,-1)[:,:nocca*nmo].reshape(3,nmo,nocca)
            mo1b = mo1.reshape(3,-1)[:,nocca*nmo:].reshape(3,nmo,noccb)
            dm1a = [reduce(numpy.dot, (mo_coeff[0], x, orboa.T.conj())) for x in mo1a]
            dm1b = [reduce(numpy.dot, (mo_coeff[1], x, orbob.T.conj())) for x in mo1b]
            dm1 = numpy.asarray(([d1-d1.conj().T for d1 in dm1a],
                                 [d1-d1.conj().T for d1 in dm1b]))
            v1ao = vresp(dm1)
            v1a = [reduce(numpy.dot, (mo_coeff[0].T.conj(), x, orboa)) for x in v1ao[0]]
            v1b = [reduce(numpy.dot, (mo_coeff[1].T.conj(), x, orbob)) for x in v1ao[1]]
            v1mo = numpy.hstack((numpy.asarray(v1a).reshape(3,-1),
                                 numpy.asarray(v1b).reshape(3,-1)))
            return v1mo.ravel()
        return vind


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom.extend([
        [1   , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ])
    mol.nucmod = {'F': 2} # gaussian nuclear model
    mol.basis = {'H': '6-31g',
                 'F': '6-31g',}
    mol.build()

    mf = scf.UHF(mol).run()
    nmr = NMR(mf)
    nmr.cphf = True
    #nmr.gauge_orig = (0,0,0)
    msc = nmr.kernel() # _xx,_yy = 375.232839, _zz = 483.002139
    print(lib.finger(msc) - -132.22894626177765)

    nmr.cphf = True
    nmr.gauge_orig = (1,1,1)
    msc = nmr.shielding()
    print(lib.finger(msc) - 4.0519712753371522)

    nmr.cphf = False
    nmr.gauge_orig = None
    msc = nmr.shielding()
    print(lib.finger(msc) - -133.26525857962628)

    mol.atom.extend([
        [1 , (1. , 0.3, .417)],
        [1 , (0.2, 1. , 0.)],])
    mol.build()
    mf = scf.UHF(mol).run()
    nmr = NMR(mf)
    nmr.cphf = False
    nmr.gauge_orig = None
    msc = nmr.shielding()
    print(lib.finger(msc) - -123.98596361883168)

