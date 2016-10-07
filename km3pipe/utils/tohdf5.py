# coding=utf-8
# Filename: tohdf5.py
"""
Convert ROOT and EVT files to HDF5.

Usage:
    tohdf5 [-o OUTFILE] [-n EVENTS] [--aa-format=<fmt>] [--aa-lib=<lib.so>] FILE...
    tohdf5 [-o OUTFILE] [-n EVENTS] [-j] [-s] [-l] FILE...
    tohdf5 (-h | --help)
    tohdf5 --version

Options:
    --aa-format=<fmt>        tohdf5: Which aanet subformat ('minidst',
                             'ancient_recolns', 'jevt_jgandalf',
                             'generic_track') [default: None]
    --aa-lib-<lib.so>        tohdf5: path to aanet binary (for old versions which
                             must be loaded via `ROOT.gSystem.Load()` instead
                             of `import aa`)
    -h --help                Show this screen.
    -j --jppy                tohdf5: Use jppy (not aanet) for Jpp readout
    -l --with-l0hits         Include L0-hits [default: False]
    -n EVENTS/RUNS           Number of events/runs.
    -o OUTFILE               Output file.
    -s --with-summaryslices  Include summary slices [default: False]
"""

from __future__ import division, absolute_import, print_function

from datetime import datetime
import os
from six import string_types
import sys

from km3modules import StatusBar
from km3pipe import version

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def tohdf5(input_files, output_file, n_events, **kwargs):
    """Convert Any file to HDF5 file"""
    from km3pipe import Pipeline  # noqa
    from km3pipe.io import GenericPump, HDF5Sink  # noqa

    pipe = Pipeline()
    pipe.attach(GenericPump, filenames=input_files, **kwargs)
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

    infiles = args['FILE']
    if isinstance(infiles, string_types):
        ofname = infiles + '.h5'
    else:
        ofname = infiles[0] + '.combined.h5'
    outfile = args['-o'] or ofname
    use_jppy_pump = args['--jppy']
    aa_format = args['--aa-format']
    aa_lib = args['--aa-lib']
    with_summaryslices = args['--with-summaryslices']
    with_l0hits = args['--with-l0hits']
    tohdf5(infiles, outfile, n, use_jppy=use_jppy_pump, aa_fmt=aa_format,
           aa_lib=aa_lib, with_summaryslices=with_summaryslices,
           with_l0hits=with_l0hits)
