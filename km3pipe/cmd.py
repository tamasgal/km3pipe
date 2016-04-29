# coding=utf-8
# Filename: cmd.py
"""
KM3Pipe command line utility.

Usage:
    km3pipe test
    km3pipe tohdf5 [-s] -i FILE -o FILE
    km3pipe (-h | --help)
    km3pipe --version

Options:
    -h --help  Show this screen.
    -i FILE    Input file.
    -o FILE    Output file.
    -s         Write each event in a separate dataset.

"""
from __future__ import division, absolute_import, print_function

from km3pipe import version


def tohdf5(input_file, output_file, use_tables=True):
    """Convert ROOT file to HDF5 file"""
    from km3pipe import Pipeline  # noqa
    from km3pipe.pumps import AanetPump, HDF5Sink, HDF5TableSink  # noqa

    sink = HDF5TableSink if use_tables else HDF5Sink

    pipe = Pipeline()
    pipe.attach(AanetPump, filename=input_file)
    pipe.attach(sink, filename=output_file)
    pipe.drain()


def main():
    from docopt import docopt
    arguments = docopt(__doc__, version=version)

    if arguments['tohdf5']:
        tohdf5(arguments['-i'], arguments['-o'], arguments['-s'])
