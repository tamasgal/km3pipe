# coding=utf-8
# Filename: root.py
# pylint: disable=C0103,R0903
# vim:set ts=4 sts=4 sw=4 et:
"""
Read and write Vanilla ROOT files.
"""
from __future__ import division, absolute_import, print_function

from six import string_types

import numpy as np
import root_numpy as rnp
from rootpy.io import root_open

from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def open_rfile(rfile, fmode='r'):
    if isinstance(rfile, string_types):
        return root_open(rfile, mode=fmode)
    return rfile


def get_hist(rfile, histname, get_overflow=False):
    """Read a 1D Histogram."""
    rfile = open_rfile(rfile)
    hist = rfile[histname]
    xlims = np.array(list(hist.xedges()))
    bin_values = rnp.hist2array(hist, include_overflow=get_overflow)
    rfile.close()
    return bin_values, xlims


def get_hist2d(rfile, histname, get_overflow=False):
    """Read a 2D Histogram."""
    rfile = open_rfile(rfile)
    hist = rfile[histname]
    xlims = np.array(list(hist.xedges()))
    ylims = np.array(list(hist.yedges()))
    bin_values = rnp.hist2array(hist, include_overflow=get_overflow)
    rfile.close()
    return bin_values, xlims, ylims


def get_hist3d(rfile, histname, get_overflow=False):
    """Read a 3D Histogram."""
    rfile = open_rfile(rfile)
    hist = rfile[histname]
    xlims = np.array(list(hist.xedges()))
    ylims = np.array(list(hist.yedges()))
    zlims = np.array(list(hist.zedges()))
    bin_values = rnp.hist2array(hist, include_overflow=get_overflow)
    rfile.close()
    return bin_values, xlims, ylims, zlims
