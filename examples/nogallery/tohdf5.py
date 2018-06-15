#!/usr/bin/env python
"""
Converts hits in a Jpp-ROOT file to HDF5.

"""
from km3pipe import Pipeline
from km3pipe.io.aanet import AanetPump
from km3pipe.io import HDF5SinkLegacy

import sys

if len(sys.argv) < 3:
    sys.exit('Usage: {0} FILENAME.root OUTPUTFILENAME.h5'.format(sys.argv[0]))

FILEPATH = sys.argv[1]
OUTPUTFILEPATH = sys.argv[2]

pipe = Pipeline()
pipe.attach(AanetPump, filename=FILEPATH)
pipe.attach(HDF5SinkLegacy, filename=OUTPUTFILEPATH)
pipe.drain()
