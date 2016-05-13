# coding=utf-8
# Filename: cmd.py
"""
KM3Pipe command line utility.

Usage:
    km3pipe test
    km3pipe tohdf5 [-n EVENTS] -i FILE -o FILE
    km3pipe h5tree [-g GROUP] -i FILE
    km3pipe (-h | --help)
    km3pipe --version

Options:
    -h --help  Show this screen.
    -i FILE    Input file.
    -o FILE    Output file.
    -n EVENTS  Number of events.
    -g GROUP   Group/Node where dataset is located [default: /]

"""
from __future__ import division, absolute_import, print_function

from km3pipe import version
from km3modules import StatusBar


def tohdf5(input_file, output_file, n_events):
    """Convert ROOT file to HDF5 file"""
    from km3pipe import Pipeline  # noqa
    from km3pipe.pumps import AanetPump, HDF5Sink  # noqa

    pipe = Pipeline()
    pipe.attach(AanetPump, filename=input_file)
    pipe.attach(StatusBar, every=1000)
    pipe.attach(HDF5Sink, filename=output_file)
    pipe.drain(n_events)


def main():
    from docopt import docopt
    arguments = docopt(__doc__, version=version)

    try:
        n_events = int(arguments['-n'])
    except TypeError:
        n_events = None

    if arguments['tohdf5']:
        tohdf5(arguments['-i'], arguments['-o'], n_events)
