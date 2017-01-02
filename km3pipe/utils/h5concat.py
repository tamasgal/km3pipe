# coding=utf-8
# Filename: h5concat.py
"""
Convert ROOT and EVT files to HDF5.

Usage:
    h5concat OUTFILE INFILES...
    h5concat (-h | --help)
    h5concat --version

Options:
    -h --help                  Show this screen.
"""

from __future__ import division, absolute_import, print_function

from km3modules import StatusBar
from km3pipe import version

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def h5concat(input_files, output_file):
    """Convert Any file to HDF5 file"""
    from km3pipe import Pipeline  # noqa
    from km3pipe.io import HDF5Pump, HDF5Sink  # noqa

    pipe = Pipeline()
    pipe.attach(HDF5Pump, filenames=input_files)
    pipe.attach(StatusBar, every=1000)
    pipe.attach(HDF5Sink, filename=output_file)
    pipe.drain()


def main():
    from docopt import docopt
    args = docopt(__doc__, version=version)

    infiles = args['INFILES']
    outfile = args['OUTFILE']

    h5concat(outfile, infiles)
