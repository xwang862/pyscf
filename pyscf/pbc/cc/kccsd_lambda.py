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
from pyscf.cc import ccsd_lambda
from pyscf.lib import logger

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

    l1new = numpy.zeros_like(l1)

    l2new = numpy.zeros_like(l2)

    return l1new, l2new