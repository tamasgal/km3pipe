# coding=utf-8
# Filename: runtable.py
"""
Prints the run table for a given detector ID.

Usage:
    runtable [-n RUNS] [-s REGEX] DET_ID
    runtable (-h | --help)
    runtable --version

Options:
    -h --help           Show this screen.
    -n RUNS             Number of runs.
    -s REGEX            Regular expression to filter the runsetup name/id.
    DET_ID              Detector ID (eg. D_ARCA001).

"""
from __future__ import division, absolute_import, print_function

import sys
import km3pipe as kp

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2018, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def runtable(det_id, n=5, sep='\t', regex=None):
    """Print the run table of the last `n` runs for given detector"""
    db = kp.db.DBManager()
    df = db.run_table(det_id)

    if regex is not None:
        df = df[df['RUNSETUPNAME'].str.match(regex) |
                df['RUNSETUPID'].str.match(regex)]

    if n is not None:
        df = df.tail(n)

    df.to_csv(sys.stdout, sep=sep)


def main():
    from docopt import docopt
    args = docopt(__doc__, version=kp.version)

    try:
        n = int(args['-n'])
    except TypeError:
        n = None

    runtable(args['DET_ID'], n, regex=args['-s'])
