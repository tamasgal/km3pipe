# Filename: __init__.py
"""
A collection of io for different kinds of data formats.

"""
from __future__ import absolute_import, print_function, division

import os.path

import numpy as np

from .evt import EvtPump    # noqa
from .daq import DAQPump    # noqa
from .clb import CLBPump    # noqa
from .aanet import AanetPump    # noqa
from .jpp import EventPump    # noqa
from .ch import CHPump    # noqa
from .hdf5 import HDF5Pump    # noqa
from .hdf5 import HDF5Sink    # noqa
from .hdf5 import HDF5MetaData    # noqa
from .pickle import PicklePump    # noqa

from km3pipe.logger import get_logger

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal, Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = get_logger(__name__)


def GenericPump(filenames, use_jppy=False, name="GenericPump", **kwargs):
    """A generic pump which utilises the appropriate pump."""
    if isinstance(filenames, str):
        filenames = [filenames]

    try:
        iter(filenames)
    except TypeError:
        log.critical("Don't know how to iterate through filenames.")
        raise TypeError("Invalid filenames.")

    extensions = set(os.path.splitext(fn)[1] for fn in filenames)

    if len(extensions) > 1:
        log.critical("Mixed filetypes, please use only files of the same type")
        raise IOError("Mixed filetypes.")

    extension = list(extensions)[0]

    io = {
        '.evt': EvtPump,
        '.h5': HDF5Pump,
        '.root': EventPump if use_jppy else AanetPump,
        '.dat': DAQPump,
        '.dqd': CLBPump,
    }

    if extension not in io:
        log.critical(
            "No pump found for file extension '{0}'".format(extension)
        )
        raise ValueError("Unknown filetype")

    missing_files = [fn for fn in filenames if not os.path.exists(fn)]
    if missing_files:
        if len(missing_files) == len(filenames):
            message = "None of the given files could be found."
            log.critical(message)
            raise SystemExit(message)
        else:
            log.warning(
                "The following files are missing and ignored: {}".format(
                    ', '.join(missing_files)
                )
            )

    input_files = set(filenames) - set(missing_files)

    if len(input_files) == 1:
        return io[extension](filename=filenames[0], name=name, **kwargs)
    else:
        return io[extension](filenames=filenames, name=name, **kwargs)


def read_calibration(
        detx=None, det_id=None, from_file=False, det_id_table=None
):
    """Retrive calibration from file, the DB."""
    from km3pipe.calib import Calibration    # noqa

    if not (detx or det_id or from_file):
        return None
    if detx is not None:
        return Calibration(filename=detx)
    if from_file:
        det_ids = np.unique(det_id_table)
        if len(det_ids) > 1:
            log.critical("Multiple detector IDs found in events.")
        det_id = det_ids[0]
    if det_id is not None:
        if det_id < 0:
            log.warning(
                "Negative detector ID found ({0}). This is a MC "
                "detector and cannot be retrieved from the DB.".format(det_id)
            )
            return None
        return Calibration(det_id=det_id)
    return None
