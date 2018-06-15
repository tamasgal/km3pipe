# Filename: root.py
# pylint: disable=C0103,R0903
# vim:set ts=4 sts=4 sw=4 et:
"""
Read and write Vanilla ROOT files.
"""
from __future__ import absolute_import, print_function, division

import numpy as np

from km3pipe.logger import get_logger

log = get_logger(__name__)    # pylint: disable=C0103

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def open_rfile(rfile, fmode='r'):
    from rootpy.io import root_open
    if isinstance(rfile, str):
        return root_open(rfile, mode=fmode)
    return rfile


def get_hist(rfile, histname, get_overflow=False):
    """Read a 1D Histogram."""
    import root_numpy as rnp

    rfile = open_rfile(rfile)
    hist = rfile[histname]
    xlims = np.array(list(hist.xedges()))
    bin_values = rnp.hist2array(hist, include_overflow=get_overflow)
    rfile.close()
    return bin_values, xlims


def get_hist2d(rfile, histname, get_overflow=False):
    """Read a 2D Histogram."""
    import root_numpy as rnp

    rfile = open_rfile(rfile)
    hist = rfile[histname]
    xlims = np.array(list(hist.xedges()))
    ylims = np.array(list(hist.yedges()))
    bin_values = rnp.hist2array(hist, include_overflow=get_overflow)
    rfile.close()
    return bin_values, xlims, ylims


def get_hist3d(rfile, histname, get_overflow=False):
    """Read a 3D Histogram."""
    import root_numpy as rnp

    rfile = open_rfile(rfile)
    hist = rfile[histname]
    xlims = np.array(list(hist.xedges()))
    ylims = np.array(list(hist.yedges()))
    zlims = np.array(list(hist.zedges()))
    bin_values = rnp.hist2array(hist, include_overflow=get_overflow)
    rfile.close()
    return bin_values, xlims, ylims, zlims


def interpol_hist2d(h2d, oversamp_factor=10):
    """Sample the interpolator of a root 2d hist.

    Root's hist2d has a weird internal interpolation routine,
    also using neighbouring bins.
    """
    from rootpy import ROOTError

    xlim = h2d.bins(axis=0)
    ylim = h2d.bins(axis=1)
    xn = h2d.nbins(0)
    yn = h2d.nbins(1)
    x = np.linspace(xlim[0], xlim[1], xn * oversamp_factor)
    y = np.linspace(ylim[0], ylim[1], yn * oversamp_factor)
    mat = np.zeros((xn, yn))
    for xi in range(xn):
        for yi in range(yn):
            try:
                mat[xi, yi] = h2d.interpolate(x[xi], y[yi])
            except ROOTError:
                continue
    return mat, x, y
