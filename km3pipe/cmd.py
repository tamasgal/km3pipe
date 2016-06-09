# coding=utf-8
# Filename: cmd.py
"""
KM3Pipe command line utility.

Usage:
    km3pipe test
    km3pipe update [BRANCH]
    km3pipe tohdf5 [-n EVENTS] -i FILE -o FILE
    km3pipe aatohdf5 [-n EVENTS] -i FILE -o FILE
    km3pipe jpptohdf5 [-n EVENTS] -i FILE -o FILE
    km3pipe evttohdf5 [-n EVENTS] -i FILE -o FILE
    km3pipe hdf2root -i FILE [-o FILE]
    km3pipe runtable [-n RUNS] DET_ID
    km3pipe (-h | --help)
    km3pipe --version

Options:
    -h --help       Show this screen.
    -i FILE         Input file.
    -o FILE         Output file.
    -n EVENTS/RUNS  Number of events/runs.
    DET_ID          Detector ID (eg. D_ARCA001).
    BRANCH          Git branch to pull (eg. develop).

"""

from __future__ import division, absolute_import, print_function

import sys
import os

from km3pipe import version
from km3pipe.db import DBManager
from km3modules import StatusBar


def aatohdf5(input_file, output_file, n_events):
    """Convert AAnet ROOT file to HDF5 file"""
    from km3pipe import Pipeline  # noqa
    from km3pipe.pumps import AanetPump, HDF5Sink  # noqa

    pipe = Pipeline()
    pipe.attach(AanetPump, filename=input_file)
    pipe.attach(StatusBar, every=1000)
    pipe.attach(HDF5Sink, filename=output_file)
    pipe.drain(n_events)


def jpptohdf5(input_file, output_file, n_events):
    """Convert JPP ROOT file to HDF5 file"""
    from km3pipe import Pipeline  # noqa
    from km3pipe.pumps import JPPPump, HDF5Sink  # noqa

    pipe = Pipeline()
    pipe.attach(JPPPump, filename=input_file)
    pipe.attach(StatusBar, every=1000)
    pipe.attach(HDF5Sink, filename=output_file)
    pipe.drain(n_events)

def evttohdf5(input_file, output_file, n_events):
    """Convert evt file to HDF5 file"""
    from km3pipe import Pipeline  # noqa
    from km3pipe.pumps import EvtPump, HDF5Sink  # noqa

    pipe = Pipeline()
    pipe.attach(EvtPump, filename=input_file)
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


def hdf2root(infile, outfile):
    from rootpy.io import root_open
    from rootpy import asrootpy
    from root_numpy import array2tree
    from tables import open_file

    h5 = open_file(infile, 'r')
    rf = root_open(outfile, 'recreate')

    # 'walk_nodes' does not allow to check if is a group or leaf
    #   exception handling is bugged
    #   introspection/typecheck is buged
    # => this moronic nested loop instead of simple `walk`
    for group in h5.walk_groups():
        for leafname, leaf in group._v_leaves.items():
            tree = asrootpy(array2tree(leaf[:], name=leaf._v_pathname))
            tree.write()
    rf.close()
    h5.close()


def update_km3pipe(git_branch):
    if git_branch == '' or git_branch is None:
        git_branch = 'master'
    os.system("pip install -U git+http://git.km3net.de/tgal/km3pipe.git@{0}"
              .format(git_branch))


def main():
    from docopt import docopt
    arguments = docopt(__doc__, version=version)

    try:
        n = int(arguments['-n'])
    except TypeError:
        n = None

    if arguments['update']:
        update_km3pipe(arguments['BRANCH'])

    if arguments['tohdf5']:
        aatohdf5(arguments['-i'], arguments['-o'], n)

    if arguments['aatohdf5']:
        aatohdf5(arguments['-i'], arguments['-o'], n)

    if arguments['jpptohdf5']:
        jpptohdf5(arguments['-i'], arguments['-o'], n)

    if arguments['evttohdf5']:
        evttohdf5(arguments['-i'], arguments['-o'], n)

    if arguments['runtable']:
        runtable(arguments['DET_ID'], n)

    if arguments['hdf2root']:
        infile = arguments['-i']
        if not arguments['-o']:
            outfile = infile + '.root'
        hdf2root(infile, outfile)
