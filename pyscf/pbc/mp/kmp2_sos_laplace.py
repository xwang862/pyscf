#!/usr/bin/env python

import numpy as np
from pyscf import lib
from pyscf.lib import logger, einsum
from pyscf.pbc.df import df
from pyscf import __config__
from pyscf.lib.parameters import LARGE_DENOM
from pyscf.pbc.mp import kmp2
from pyscf.pbc.mp.kmp2 import padding_k_idx, _init_mp_df_eris

import scipy
from scipy import integrate, optimize


WITH_T2 = getattr(__config__, "mp_mp2_with_t2", True)


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


def get_weights_haser1992(eia, tau=7):
    """Get weights and grids for Laplace transform using least square fit by Almlof & Haser.
    Ref: Marco Häser and Jan Almlöf, J. Chem. Phys. 96, 489 (1992)
    Note 1: I use Levenberg–Marquardt (LM) least-squares algorithm (implemented in scipy.optimize.least_squares)
    instead of Simplex algorithm. LM was also used by
    Doser, Lambrecht, Kussmann, and Ochsenfeld, J. Chem. Phys. 130, 064107 (2009)
    Note 2: An implementation of Nelder-Mead Simplex algorithm (for non-linear functions) can be found here:
    https://github.com/gyrov/Simplex_Algorithm . Ref:
    Nelder and Mead, A Simplex Method for Function Minimization, The Computer Journal, 7, 308 (1965)

    Args:
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

    ts = opt_result.x
    ws = get_ws(ts, interval=interval, ngrid=ngrid)
    return ws, ts


def laplace_transform(denom, weights, grids):
    """Laplace transform of 1/x.
    If x < 0, then 1/x ~= - sum_i weights[i] * exp(grids[i] * x)

    Args:
        denom (ndarray): denominator in 1/x
        weights (1d array): weights for each grid point
        grids (1d array): grid points

    Returns:
        ndarray: evaluated 1/x, with same shape as denom
    """
    return -np.sum(
        [w * np.exp(t * denom) for t, w in zip(grids, weights)], axis=0
    )


def kernel(
    mp,
    mo_energy,
    mo_coeff,
    cos=1.36,
    verbose=logger.NOTE,
    with_t2=False,
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
        cos (float, optional): scaling coefficient for SOS-MP2. Defaults to 1.36.
        verbose (int, optional): level of verbosity. Defaults to logger.NOTE (=3).
        with_t2 (bool, optional): whether to compute t2 amplitudes. Defaults to False.
        method (int, optional): Laplace transform method to compute SOS-MP2 energy. Defaults to 3 (least square fit by Haser & Almlof).
        tau (int, optional): Number of quadrature points. Defaults to 3.

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
    if not with_df_ints:
        raise NotImplementedError(
            "O(N^4) SOS-KMP2 is only implemented for GDF integrals."
        )

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

    kconserv = mp.khelper.kconserv

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind="split")

    if with_t2:
        raise NotImplementedError(
            "t2 amplitudes are not implemented for O(N^4) algorithm."
        )
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

    if method != 3:
        raise NotImplementedError(
            f"Only method 3 ({method_name[method]}) is implemented."
        )

    emp2_ss = emp2_os = 0.0
    for ki in range(nkpts):
        for kj in range(nkpts):
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

                # Get weights and grids for laplace transform
                ws, ts = get_weights_haser1992(eia, tau)
                # For each quad point t,
                # M_PQ = L_ia^P.conj * L_ia^Q * exp((ei-ea)*t)
                # N_PQ = L_jb^P.conj * L_jb^Q * exp((ej-eb)*t)
                mtensor = (1.0 / nkpts) * np.array(
                    [
                        einsum(
                            "Pia,Qia,ia->PQ",
                            Lov[ki, ka].conj(),
                            Lov[ki, ka],
                            np.exp(eia * t),
                        )
                        for t in ts
                    ]
                )
                ntensor = (1.0 / nkpts) * np.array(
                    [
                        einsum(
                            "Pjb,Qjb,jb->PQ",
                            Lov[kj, kb].conj(),
                            Lov[kj, kb],
                            np.exp(ejb * t),
                        )
                        for t in ts
                    ]
                )
                emp2_os += -einsum("q,qPQ,qPQ", ws, mtensor, ntensor).real

    log.timer("KMP2", *cput0)

    emp2_os /= nkpts
    emp2 = lib.tag_array(cos * emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2, t2


class LaplaceSOSKMP2(kmp2.KMP2):
    def kernel(
        self, mo_energy=None, mo_coeff=None, with_t2=False, method=3, tau=7
    ):
        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None:
            logger.warn(
                "mo_coeff, mo_energy are not given.\n"
                "You may need to call mf.kernel() to generate them."
            )
            raise RuntimeError

        mo_coeff, mo_energy = kmp2._add_padding(self, mo_coeff, mo_energy)

        # TODO: compute e_hf for non-canonical SCF
        self.e_hf = self._scf.e_tot

        self.e_corr, self.t2 = kernel(
            self,
            mo_energy,
            mo_coeff,
            verbose=self.verbose,
            with_t2=with_t2,
            method=method,
            tau=tau,
        )

        self.e_corr_ss = getattr(self.e_corr, "e_corr_ss", 0)
        self.e_corr_os = getattr(self.e_corr, "e_corr_os", 0)
        self.e_corr = float(self.e_corr)

        self._finalize()

        return self.e_corr, self.t2

    def _finalize(self):
        """Hook for dumping results and clearing up the object"""
        log = logger.new_logger(self)
        log.info("E_corr(oppo-spin) = %.15g", self.e_corr_os)
        return self


if __name__ == "__main__":
    from pyscf.pbc import gto, scf

    cell = gto.Cell()
    cell.atom = """
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    """
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.a = """
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000"""
    cell.unit = "B"
    cell.verbose = 5
    cell.build()

    kpts = cell.make_kpts([2, 2, 2])
    kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
    ehf = kmf.kernel()

    mymp = LaplaceSOSKMP2(kmf)
    esosmp2 = mymp.kernel(with_t2=False, method=3, tau=7)[0]

    emp2_ref = -0.13314158977189
    print(f"Laplace-SOS-MP2: {esosmp2:.12f}\nMP2: {emp2_ref:.12f}")
