# coding=utf-8
# Filename: __init__.py
"""
A collection of pumps for different kinds of data formats.

"""
from __future__ import division, absolute_import, print_function

import os

from km3pipe.pumps.evt import EvtPump  # noqa
from km3pipe.pumps.daq import DAQPump  # noqa
from km3pipe.pumps.clb import CLBPump  # noqa
from km3pipe.pumps.aanet import AanetPump  # noqa
from km3pipe.pumps.jpp import JPPPump  # noqa
from km3pipe.pumps.ch import CHPump  # noqa
from km3pipe.pumps.hdf5 import HDF5Pump  # noqa
from km3pipe.pumps.hdf5 import HDF5Sink  # noqa
from km3pipe.pumps.pickle import PicklePump  # noqa

from km3pipe.logger import logging

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


log = logging.getLogger(__name__)


def GenericPump(filename, use_jppy=False, name="GenericPump"):
    """A generic pump which utilises the appropriate pump."""
    extension = os.path.splitext(filename)[1]

    pumps = {
            '.evt': EvtPump,
            '.h5': HDF5Pump,
            '.aa.root': AanetPump,
            '.root': JPPPump if use_jppy else AanetPump,
            '.dat': DAQPump,
            '.dqd': CLBPump,
            }

    if extension not in pumps:
        log.critical("No pump found for '{0}'".format(extension))

    return pumps[extension](filename=filename, name=name)


def df_to_h5(df, filename, tabname, filemode='a', where='/', complevel=5,):
    """Write pandas dataframes with proper columns.

    The main 2 ways pandas writes dataframes suck bigly.
    """
    from tables import Filters, open_file
    with open_file(filename, filemode) as h5:
        filt = Filters(complevel=complevel, shuffle=True)
        h5.create_table(where, tabname, obj=df.to_records(), filters=filt)
