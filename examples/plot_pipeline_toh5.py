#!/usr/bin/env python
"""
======================
ToHDF5 Pipeline example
======================

"""

# Authors: Moritz Lotze <mlotze@km3net.de>
# License: BSD-3
# Date: 2018-07-12
# Status: Under construction...

from __future__ import absolute_import, print_function, division

import tables as tb

from km3pipe import Pipeline
from km3pipe.io import EvtPump, HDF5Sink

from km3modules.common import StatusBar

#####################################################
# Preparation
# -----------
# Let's define the inputs / outputs first -- those would be coming from a CLI
# parser in practice.

N_EVENTS = 200000
IN_FNAME = 'data/numu_cc.evt'
OUT_FNAME = 'data/numu_cc.h5'

#####################################################
# Also, in this case we don't really want to dump the data onto disk, so we
# create an in-memory-ony file, and pass it as the ``h5file`` arg to the
# hdf5sink. to actually write out a file, just specify an outfile name
# (commented out here).

OUTFILE = tb.open_file(
    # create the file in memory only
    OUT_FNAME,
    'w',
    driver="H5FD_CORE",
    driver_core_backing_store=0,
)

#####################################################
# Setting up the pipeline
# -----------------------

pipe = Pipeline(timeit=True)
pipe.attach(EvtPump, filename=IN_FNAME)
pipe.attach(StatusBar, every=25)
pipe.attach(
    HDF5Sink,
    # filename=OUT_FNAME,
    h5file=OUTFILE,
)

#####################################################
# Draining the pipeline
# ---------------------

pipe.drain(N_EVENTS)
