#!/usr/bin/env python

import os
import numpy as np
from pyscf import lib
from pyscf.lib import logger, einsum
from pyscf.pbc.df import df
from pyscf import __config__
from pyscf.lib.parameters import LARGE_DENOM
from pyscf.pbc.mp.kmp2 import padding_k_idx, _init_mp_df_eris, _add_padding

import scipy
from scipy import integrate, optimize


WITH_T2 = getattr(__config__, "mp_mp2_with_t2", True)

# Laplace transform followed by logarithmic transform
def laplace_ayala2001(eijab, eia, tau=7):
    """Laplace transform followed by logarithmic transform.
    Ref: Ayala, Kudin, and Scuseria. J. Chem. Phys. 115, 9698 (2001)

    Args:
        eijab (numpy array): ei + ej - ea - eb
        eia (numpy array): ei - ea
        tau (int, optional): Number of quadrature points. Defaults to 7.

    Returns:
        ndarray: 1/e_ijab
    """
    nocc, nvir = eia.shape
    alpha = -2.0 * eia[nocc - 1, 0]

    def laplace(t):
        if isinstance(t, (list, tuple, np.ndarray)):
            return np.array(
                [np.power(x, -eijab / alpha - 1.0) for x in t]
            ).transpose(1, 2, 3, 4, 0)
        else:
            return np.power(t, -eijab / alpha - 1.0)

    inv_e_ijab = -1.0 / alpha * integrate.fixed_quad(laplace, 0, 1, n=tau)[0]
    return inv_e_ijab


def laplace_haser1992(eijab, eia, tau=7):
    """Least square fit by Almlof & Haser. 
    Ref: Marco Häser and Jan Almlöf, J. Chem. Phys. 96, 489 (1992)
    Note 1: I use Levenberg–Marquardt (LM) least-squares algorithm (implemented in scipy.optimize.least_squares)
    instead of Simplex algorithm. LM was also used by
    Doser, Lambrecht, Kussmann, and Ochsenfeld, J. Chem. Phys. 130, 064107 (2009)
    Note 2: An implementation of Nelder-Mead Simplex algorithm (for non-linear functions) can be found here:
    https://github.com/gyrov/Simplex_Algorithm . Ref:
    Nelder and Mead, A Simplex Method for Function Minimization, The Computer Journal, 7, 308 (1965)

    Args:
        eijab (numpy array): ei + ej - ea - eb
        eia (numpy array): ei - ea
        tau (int, optional): Number of quadrature points. Defaults to 7.

    Returns:
        ndarray: 1/e_ijab
    """

    def get_log_grids(interval, ngrid):
        emin, emax = interval
        grids = np.logspace(np.log10(emin), np.log10(emax), ngrid + 1)
        xs = (grids[1:] + grids[:-1]) / 2
        dxs = np.diff(grids)
        return xs, dxs

    def get_ws(ts, interval=(0.1, 1.0), ngrid=10):
        xs, dxs = get_log_grids(interval, ngrid)

        # Assume equal distribution of e_ijab over the interval
        fx = np.ones(ngrid)
        # Solve linear equations B.w=a for w's, given t's
        a_vec = np.array([np.sum(dxs * fx / xs * np.exp(-t * xs)) for t in ts])
        B_mat = np.array(
            [
                [np.sum(dxs * fx * np.exp(-(ti + tj) * xs)) for tj in ts]
                for ti in ts
            ]
        )
        ws = scipy.linalg.solve(B_mat, a_vec)
        return ws

    def residual(ts, interval=(0.1, 1.0), ngrid=10):
        xs, dxs = get_log_grids(interval, ngrid)

        # Assume equal distribution of e_ijab over the interval
        fx = np.ones(ngrid)
        # Get weights
        ws = get_ws(ts, interval, ngrid)
        # Get approximate 1/x
        approx = np.sum(
            [w * np.exp(-t * xs) for w, t in zip(ws, ts)],
            axis=0,
        )
        # Since scipy.optimize.least_squares basically minimizes sum(f_i(x)**2), where f_i is residual function,
        # we don't need to perform `sum` or `square` here.
        return np.sqrt(dxs * fx * np.square(1.0 / xs - approx))

    t0s = np.linspace(0, 1, tau)
    nocc, nvir = eia.shape
    emin, emax = (
        2.0 * np.abs(eia[nocc - 1, 0]),
        2.0 * np.abs(eia[0, nvir - 1]),
    )
    interval = (emin, emax)
    ngrid = 1200

    opt_result = optimize.least_squares(
        residual,
        t0s,
        kwargs={"interval": interval, "ngrid": ngrid},
    )

    # print(f"fitting error: {opt_result.cost}")

    ts = opt_result.x
    ws = get_ws(ts, interval=interval, ngrid=ngrid)
    # print(f"ts: {ts}")
    # print(f"ws: {ws}")

    inv_e_ijab = -np.sum(
        [w * np.exp(t * eijab) for t, w in zip(ts, ws)], axis=0
    )
    return inv_e_ijab


def kernel(
    mp,
    mo_energy,
    mo_coeff,
    verbose=logger.NOTE,
    with_t2=WITH_T2,
    method=3,
    tau=3,
):
    """Computes k-point RMP2 energy.

    Args:
        mp (KMP2): an instance of KMP2
        mo_energy (list): a list of numpy.ndarray. Each array contains MO energies of
                          shape (Nmo,) for one kpt
        mo_coeff (list): a list of numpy.ndarray. Each array contains MO coefficients
                         of shape (Nao, Nmo) for one kpt
        verbose (int, optional): level of verbosity. Defaults to logger.NOTE (=3).
        with_t2 (bool, optional): whether to compute t2 amplitudes. Defaults to WITH_T2 (=True).

    Returns:
        KMP2 energy and t2 amplitudes (=None if with_t2 is False)
    """
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp, verbose)

    mp.dump_flags()
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts

    with_df_ints = mp.with_df_ints and isinstance(mp._scf.with_df, df.GDF)

    mem_avail = mp.max_memory - lib.current_memory()[0]
    mem_usage = (nkpts * (nocc * nvir) ** 2) * 16 / 1e6
    if with_df_ints:
        mydf = mp._scf.with_df
        if mydf.auxcell is None:
            # Calculate naux based on precomputed GDF integrals
            naux = mydf.get_naoaux()
        else:
            naux = mydf.auxcell.nao_nr()

        mem_usage += (nkpts**2 * naux * nocc * nvir) * 16 / 1e6
    if with_t2:
        mem_usage += (nkpts**3 * (nocc * nvir) ** 2) * 16 / 1e6
    if mem_usage > mem_avail:
        raise MemoryError(
            "Insufficient memory! MP2 memory usage %d MB (currently available %d MB)"
            % (mem_usage, mem_avail)
        )

    eia = np.zeros((nocc, nvir))
    eijab = np.zeros((nocc, nocc, nvir, nvir))

    fao2mo = mp._scf.with_df.ao2mo
    kconserv = mp.khelper.kconserv
    oovv_ij = np.zeros(
        (nkpts, nocc, nocc, nvir, nvir), dtype=mo_coeff[0].dtype
    )

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind="split")

    if with_t2:
        t2 = np.zeros(
            (nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=complex
        )
    else:
        t2 = None

    # Build 3-index DF tensor Lov
    if with_df_ints:
        Lov = _init_mp_df_eris(mp)

    method_name = {
        0: "Reference method",
        2: "Laplace transform followed by logarithmic transform",
        3: "Least square fit by Almlof \& Haser",
    }
    print(f"\nLaplace method: {method_name[method]}\ntau: {tau}\n")

    emp2_ss = emp2_os = 0.0
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                # (ia|jb)
                if with_df_ints:
                    oovv_ij[ka] = (1.0 / nkpts) * einsum(
                        "Lia,Ljb->iajb", Lov[ki, ka], Lov[kj, kb]
                    ).transpose(0, 2, 1, 3)
                else:
                    orbo_i = mo_coeff[ki][:, :nocc]
                    orbo_j = mo_coeff[kj][:, :nocc]
                    orbv_a = mo_coeff[ka][:, nocc:]
                    orbv_b = mo_coeff[kb][:, nocc:]
                    oovv_ij[ka] = (
                        fao2mo(
                            (orbo_i, orbv_a, orbo_j, orbv_b),
                            (
                                mp.kpts[ki],
                                mp.kpts[ka],
                                mp.kpts[kj],
                                mp.kpts[kb],
                            ),
                            compact=False,
                        )
                        .reshape(nocc, nvir, nocc, nvir)
                        .transpose(0, 2, 1, 3)
                        / nkpts
                    )
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]

                # Remove zero/padded elements from denominator
                eia = LARGE_DENOM * np.ones(
                    (nocc, nvir), dtype=mo_energy[0].dtype
                )
                n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
                eia[n0_ovp_ia] = (mo_e_o[ki][:, None] - mo_e_v[ka])[n0_ovp_ia]

                ejb = LARGE_DENOM * np.ones(
                    (nocc, nvir), dtype=mo_energy[0].dtype
                )
                n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
                ejb[n0_ovp_jb] = (mo_e_o[kj][:, None] - mo_e_v[kb])[n0_ovp_jb]

                eijab = lib.direct_sum("ia,jb->ijab", eia, ejb)

                ### Start of Laplace
                if method == 0:
                    # Reference method
                    inv_e_ijab = 1.0 / eijab
                elif method == 2:
                    inv_e_ijab = laplace_ayala2001(eijab, eia, tau)
                elif method == 3:
                    inv_e_ijab = laplace_haser1992(eijab, eia, tau)

                t2_ijab = np.conj(oovv_ij[ka] * inv_e_ijab)
                if with_t2:
                    t2[ki, kj, ka] = t2_ijab
                edi = einsum("ijab,ijab", t2_ijab, oovv_ij[ka]).real * 2
                exi = -einsum("ijab,ijba", t2_ijab, oovv_ij[kb]).real
                emp2_ss += edi * 0.5 + exi
                emp2_os += edi * 0.5

    log.timer("KMP2", *cput0)

    emp2_ss /= nkpts
    emp2_os /= nkpts
    emp2 = lib.tag_array(
        emp2_ss + emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os
    )

    return emp2, t2


class LaplaceKMP2(kmp2.KMP2):
    def kernel(self, mo_energy=None, mo_coeff=None, with_t2=WITH_T2, method=3, tau=7):
        if mo_energy is None: mo_energy = self.mo_energy
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None:
            logger.warn('mo_coeff, mo_energy are not given.\n'
                        'You may need to call mf.kernel() to generate them.')
            raise RuntimeError

        mo_coeff, mo_energy = kmp2._add_padding(self, mo_coeff, mo_energy)

        # TODO: compute e_hf for non-canonical SCF
        self.e_hf = self._scf.e_tot

        self.e_corr, self.t2 = \
                kernel(self, mo_energy, mo_coeff, verbose=self.verbose, with_t2=with_t2, method=method, tau=tau)

        self.e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        self.e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        self.e_corr = float(self.e_corr)

        self._finalize()
        
        return self.e_corr, self.t2

def main():
    from pyscf.pbc import gto, scf, mp
    from pyscf.pbc.tools import lattice, pyscf_ase

    # Create a Cell object
    cell = gto.Cell()
    cell.unit = "B"
    ase_atom = lattice.get_ase_atom("c")
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.verbose = 3
    # cell.max_memory = 500000
    cell.build()

    # KHF
    nk = [2, 2, 2]
    kpts = cell.make_kpts(nk)
    kmf = scf.KRHF(cell, kpts).density_fit()
    # save ERIs if not exist
    h5name = "./cderi-c-szv-222.h5"
    kmf.with_df._cderi_to_save = h5name
    if os.path.isfile(h5name):
        kmf.with_df._cderi = h5name
    kmf.kernel()

    # reference KMP2
    kmp = mp.KMP2(kmf)
    emp2_ref = kmp.kernel()[0]

    energies = []
    methods = [2, 3]
    taus = range(3, 8)
    # laplace MP2
    mo_coeff, mo_energy = _add_padding(kmp, kmp.mo_coeff, kmp.mo_energy)
    for method in methods:
        for tau in taus:
            emp2_new = kernel(
                kmp, mo_energy, mo_coeff, with_t2=False, method=method, tau=tau
            )[0]
            diff = emp2_new - emp2_ref
            energies.append([method, tau, emp2_ref, emp2_new, diff])

    print(f"\n{'Method':>8} {'Tau':>8} {'EMP2_ref':>20} {'EMP2_new':>20} {'Diff':>20}")
    for method in methods:
        for tau in taus:
            method, tau, emp2_ref, emp2_new, diff = energies.pop(0)
            print(f"{method:>8} {tau:>8} {emp2_ref:>20.12f} {emp2_new:>20.12f} {diff:>20.12f}")



if __name__ == "__main__":
    main()
