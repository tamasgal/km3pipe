# coding=utf-8
# Filename: root.py
# pylint: disable=C0103,R0903
# vim:set ts=4 sts=4 sw=4 et:
"""
Read and write Vanilla ROOT files.
"""
from __future__ import division, absolute_import, print_function

from collections import defaultdict
from six import string_types

import numpy as np
import rootpy as rp
import root_numpy as rnp
import rootpy.ROOT as ROOT
from rootpy.io import root_open
from rootpy.plotting import Hist, Hist2D, Hist3D
from rootpy.tree import Tree, TreeModel, FloatCol, IntCol

import km3pipe as kp
from km3pipe import Pump, Module
from km3pipe.dataclasses import KM3Array, deserialise_map
from km3pipe.logger import logging
from km3pipe.tools import camelise, decamelise, insert_prefix_to_dtype, split

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
    #bin_errors = list(hist.xerror())
    rfile.close()
    return bin_values, xlims#, bin_errors


def get_hist2d(rfile, histname, get_overflow=False):
    """Read a 2D Histogram."""
    rfile = open_rfile(rfile)
    hist = rfile[histname]
    xlims = np.array(list(hist.xedges()))
    ylims = np.array(list(hist.yedges()))
    bin_values = rnp.hist2array(hist, include_overflow=get_overflow)
    #bin_errors = list(hist.xerror())
    rfile.close()
    return bin_values, xlims, ylims#, bin_errors


def get_hist3d(rfile, histname, get_overflow=False):
    """Read a 3D Histogram."""
    rfile = open_rfile(rfile)
    hist = rfile[histname]
    xlims = np.array(list(hist.xedges()))
    ylims = np.array(list(hist.xedges()))
    zlims = np.array(list(hist.zedges()))
    bin_values = rnp.hist2array(hist, include_overflow=get_overflow)
    #bin_errors = list(hist.xerror())
    rfile.close()
    return bin_values, xlims, ylims, zlims#, bin_errors
