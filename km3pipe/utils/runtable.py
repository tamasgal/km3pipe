# Filename: runtable.py
"""
Prints the run table for a given detector ID.

Usage:
    runtable [options] DET_ID
    runtable (-h | --help)
    runtable --version

Options:
    -h --help           Show this screen.
    -c                  Compact view.
    -n RUNS             Number of runs.
    -r FROM_RUN-TO_RUN  Range of runs (example: 3100-3200).
    -s REGEX            Regular expression to filter the runsetup name/id.
    DET_ID              Detector ID (eg. D_ARCA001).

"""

import re
import sys
import km3pipe as kp

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2018, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = kp.logger.get_logger(__name__)


def runtable(det_id, n=5, run_range=None, compact=False, sep='\t', regex=None):
    """Print the run table of the last `n` runs for given detector"""
    db = kp.db.DBManager()
    df = db.run_table(det_id)

    if run_range is not None:
        try:
            from_run, to_run = [int(r) for r in run_range.split('-')]
        except ValueError:
            log.critical("Please specify a valid range (e.g. 3100-3200)!")
            raise SystemExit
        else:
            df = df[(df.RUN >= from_run) & (df.RUN <= to_run)]

    if regex is not None:
        try:
            re.compile(regex)
        except re.error:
            log.error("Invalid regex!")
            return

        df = df[df['RUNSETUPNAME'].str.contains(regex)
                | df['RUNSETUPID'].str.contains(regex)]

    if n is not None:
        df = df.tail(n)

    if compact:
        df = df[['RUN', 'DATETIME', 'RUNSETUPNAME']]

    df.to_csv(sys.stdout, sep=sep)


def main():
    from docopt import docopt
    args = docopt(__doc__, version=kp.version)

    try:
        n = int(args['-n'])
    except TypeError:
        n = None

    runtable(
        args['DET_ID'],
        n=n,
        run_range=args['-r'],
        regex=args['-s'],
        compact=args['-c']
    )
