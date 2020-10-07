#!usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: stats.py
# pylint: disable=C0103
"""
Statistics.
"""
import numpy as np

from .logger import get_logger

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2017, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"

log = get_logger(__name__)  # pylint: disable=C0103


def mad(v):
    """MAD -- Median absolute deviation. More robust than standard deviation."""
    return np.median(np.abs(v - np.median(v)))


def mad_std(v):
    """Robust estimate of standard deviation using the MAD.

    Lifted from astropy.stats."""
    MAD = mad(v)
    return MAD * 1.482602218505602


def drop_zero_variance(df):
    """Remove columns from dataframe with zero variance."""
    return df.copy().loc[:, df.var() != 0].copy()


def perc(arr, p=95, **kwargs):
    """Create symmetric percentiles, with ``p`` coverage."""
    offset = (100 - p) / 2
    return np.percentile(arr, (offset, 100 - offset), **kwargs)


def bincenters(bins):
    """Bincenters, assuming they are all equally spaced."""
    bins = np.atleast_1d(bins)
    return 0.5 * (bins[1:] + bins[:-1])
