#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:set ts=4 sts=4 sw=4 et:
"""
Functions and modules to work with hits.
"""
from __future__ import absolute_import, print_function, division

import numpy as np

import km3pipe as kp

log = kp.logger.get_logger(__name__)
try:
    from numba import jit
except ImportError:
    log.warning(
        "This module requires `numba` to be installed, otherwise "
        "the functions and Modules imported from this module can "
        "be painfully slow."
    )
    jit = lambda f: f

__author__ = "Tamas Gal"


@jit
def count_multiplicities(times, tmax=20):
    """Calculate an array of multiplicities and corresponding coincidence IDs

    Note that this algorithm does not take care about DOM IDs, so it has to
    be fed with DOM hits.

    Parameters
    ----------
    times: array[float], shape=(n,)
        Hit times for n hits
    dt: int [default: 20]
        Time window of a coincidence

    Returns
    -------
    (array[int]), array[int]), shape=(n,)

    """
    n = times.shape[0]
    mtp = np.ones(n, dtype='<i4')    # multiplicities
    cid = np.zeros(n, '<i4')    # coincidence id
    idx0 = 0
    _mtp = 1
    _cid = 0
    t0 = times[idx0]
    for i in range(1, n):
        dt = times[i] - t0
        if dt > tmax:
            mtp[idx0:i] = _mtp
            cid[idx0:i] = _cid
            _mtp = 0
            _cid += 1
            idx0 = i
            t0 = times[i]
        _mtp += 1
        if i == n - 1:
            mtp[idx0:] = _mtp
            cid[idx0:] = _cid
            break

    return mtp, cid
