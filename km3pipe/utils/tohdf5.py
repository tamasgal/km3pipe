# coding=utf-8
# Filename: tohdf5.py
"""
Convert ROOT and EVT files to HDF5.

Usage:
    tohdf5 FILE [-o OUTFILE] [-n EVENTS] [--aa-format=<fmt>] [--aa-lib=<lib.so>]
    tohdf5 FILE [-o OUTFILE] [-n EVENTS] [-j] [-s] [-l]
    tohdf5 (-h | --help)
    tohdf5 --version

Options:
    --aa-format=<fmt>       tohdf5: Which aanet subformat ('minidst',
                            'jevt_jgandalf', 'generic_track') [default: None]
    --aa-lib-<lib.so>       tohdf5: path to aanet binary (for old versions which
                            must be loaded via `ROOT.gSystem.Load()` instead
                            of `import aa`)
    -h --help               Show this screen.
    -j --jppy               tohdf5: Use jppy (not aanet) for Jpp readout
    -l --with-l0hits        Include L0-hits [default: False]
    -n EVENTS/RUNS          Number of events/runs.
    -o OUTFILE              Output file.
    -s --with-summarslices  Include summary slices [default: False]
"""

from __future__ import division, absolute_import, print_function

import sys
import os
from datetime import datetime

from km3pipe import version
from km3modules import StatusBar

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def tohdf5(input_file, output_file, n_events, **kwargs):
    """Convert Any file to HDF5 file"""
    from km3pipe import Pipeline  # noqa
    from km3pipe.io import GenericPump, HDF5Sink  # noqa

    pipe = Pipeline()
    pipe.attach(GenericPump, filename=input_file, **kwargs)
    pipe.attach(StatusBar, every=1000)
    pipe.attach(HDF5Sink, filename=output_file)
    pipe.drain(n_events)


def main():
    from docopt import docopt
    args = docopt(__doc__, version=version)

    try:
        n = int(args['-n'])
    except TypeError:
        n = None

    infile = args['FILE']
    outfile = args['-o'] or infile + '.h5'
    use_jppy_pump = args['--jppy']
    aa_format = args['--aa-format']
    aa_lib = args['--aa-lib']
    with_summaryslices = args['--with-summarslices']
    with_l0hits = args['--with-l0hits']
    tohdf5(infile, outfile, n, use_jppy=use_jppy_pump, aa_fmt=aa_format,
           aa_lib=aa_lib, with_summaryslices=with_summaryslices,
           with_l0hits=with_l0hits)
