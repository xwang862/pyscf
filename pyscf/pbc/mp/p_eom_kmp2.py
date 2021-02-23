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

from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)  # noqa

einsum = lib.einsum

def ee_matvec_singlet(eom, nroots=1, koopmans=False, guess=None, left=False,
                      eris=None, imds=None, diag=None, partition='mp',
                      kptlist=None, dtype=None):
    """See `eom_kgccsd.kernel()` for a description of arguments. 
    """
    eom.converged, eom.e, eom.v  \
            = eom_kgccsd.kernel_ee(eom, nroots, koopmans, guess, left, eris=eris,
                                   imds=imds, diag=diag, partition=partition,
                                   kptlist=kptlist, dtype=dtype)
    return eom.e, eom.v


class PEOMMP2EESinglet(eom_krccsd.EOMEESinglet):
    """P–EOM–MBPT2 class where
    T1 = 0, 
    T2 = 1st-order t2 amplitudes, and
    doubles-doubles block of Hbar = \epsilon_a + \epsilon_b - \epsilon_i - \epsilon_j (0th-order effective Hamiltonian).

    Ref: Joshua J. Goings, Marco Caricato, Michael J. Frisch, and Xiaosong Li, J. Chem. Phys. 141, 164116 (2014)
    """
    def __init__(self, mp):
        eom_krccsd.EOMEESinglet.__init__(self, mp)

    matvec = ee_matvec_singlet

