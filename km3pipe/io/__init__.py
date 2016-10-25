# coding=utf-8
# Filename: __init__.py
"""
A collection of io for different kinds of data formats.

"""
from __future__ import division, absolute_import, print_function

import os.path
from six import string_types

import numpy as np

from km3pipe import Geometry
from km3pipe.io.evt import EvtPump  # noqa
from km3pipe.io.daq import DAQPump  # noqa
from km3pipe.io.clb import CLBPump  # noqa
from km3pipe.io.aanet import AanetPump  # noqa
from km3pipe.io.jpp import JPPPump  # noqa
from km3pipe.io.ch import CHPump  # noqa
from km3pipe.io.hdf5 import HDF5Pump  # noqa
from km3pipe.io.hdf5 import HDF5Sink  # noqa
from km3pipe.io.pickle import PicklePump  # noqa
from km3pipe.tools import insert_prefix_to_dtype
from km3pipe.io.pandas import (H5Chain, df_to_h5, map2df, load_mva, open_hdf5,
                               read_hdf5, read_group, read_table, write_table,)


from km3pipe.logger import logging

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal, Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


log = logging.getLogger(__name__)


def GenericPump(filenames, use_jppy=False, name="GenericPump", **kwargs):
    """A generic pump which utilises the appropriate pump."""
    if not isinstance(filenames, string_types):
        fn = filenames[0]
    else:
        fn = filenames
    extension = os.path.splitext(fn)[1]

    io = {
        '.evt': EvtPump,
        '.h5': HDF5Pump,
        '.root': JPPPump if use_jppy else AanetPump,
        '.dat': DAQPump,
        '.dqd': CLBPump,
    }

    if extension not in io:
        log.critical("No pump found for '{0}'".format(extension))

    if isinstance(filenames, string_types):
        return io[extension](filename=filenames, name=name, **kwargs)
    else:
        if len(filenames) == 1:
            return io[extension](filename=filenames[0], name=name, **kwargs)
        return io[extension](filenames=filenames, name=name, **kwargs)


def read_geometry(detx=None, det_id=None, from_file=False, det_id_table=None):
    """Retrive geometry from file, the DB."""
    if not detx or det_id or from_file:
        return None
    if detx is not None:
        return Geometry(filename=detx)
    if from_file:
        det_ids = np.unique(det_id_table)
        if len(det_ids) > 1:
            log.critical("Multiple detector IDs found in events.")
        det_id = det_ids[0]
    if det_id is not None:
        if det_id < 0:
            log.warning("Negative detector ID found ({0}), skipping..."
                        .format(det_id))
            return None
        try:
            return Geometry(det_id=det_id)
        except ValueError:
            log.warning("Could not retrieve the geometry information.")
    return None
