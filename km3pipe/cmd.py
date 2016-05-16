# coding=utf-8
# Filename: cmd.py
"""
KM3Pipe command line utility.

Usage:
    km3pipe test
    km3pipe tohdf5 [-n EVENTS] -i FILE -o FILE
    km3pipe runtable [-n RUNS] DET_ID
    km3pipe (-h | --help)
    km3pipe --version

Options:
    -h --help       Show this screen.
    -i FILE         Input file.
    -o FILE         Output file.
    -n EVENTS/RUNS  Number of events/runs.
    DET_ID          Detector ID (eg. D_ARCA001).

"""
from __future__ import division, absolute_import, print_function

import sys

from km3pipe import version
from km3pipe.db import DBManager
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


def runtable(det_id, n=5, sep='\t'):
    """Print the run table of the last `n` runs for given detector"""
    db = DBManager()
    df = db.run_table(det_id)
    if n is None:
        selected_df = df
    else:
        selected_df = df.tail(n)
    selected_df.to_csv(sys.stdout, sep=sep)


def main():
    from docopt import docopt
    arguments = docopt(__doc__, version=version)

    try:
        n = int(arguments['-n'])
    except TypeError:
        n = None

    if arguments['tohdf5']:
        tohdf5(arguments['-i'], arguments['-o'], n)

    if arguments['runtable']:
        runtable(arguments['DET_ID'], n)
