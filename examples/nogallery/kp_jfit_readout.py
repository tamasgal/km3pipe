#!/usr/bin/env python
"""
Usage:
    kp_jfit_readout.py FILENAME
"""

from docopt import docopt

from km3pipe.io.hdf5 import HDF5Sink
from km3pipe.io.jpp import FitPump
from km3pipe import Pipeline


def print_fits(blob):
    fits = blob['JFit']
    print(fits[:10])


if __name__ == '__main__':
    args = docopt(__doc__)
    fname = args['FILENAME']

    pipe = Pipeline()
    pipe.attach(FitPump, filename=fname)
    pipe.attach(print_fits)
    pipe.attach(HDF5Sink, filename=fname + '.h5')
    pipe.drain(1)
