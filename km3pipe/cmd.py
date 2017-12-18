# coding=utf-8
# Filename: cmd.py
"""
KM3Pipe command line utility.

Usage:
    km3pipe test
    km3pipe update [GIT_BRANCH]
    km3pipe createconf [--overwrite] [--dump]
    km3pipe detx DET_ID [-m] [-t T0_SET] [-c CALIBR_ID] [-o OUTFILE]
    km3pipe detectors [-s REGEX] [--temporary]
    km3pipe runtable [-n RUNS] [-s REGEX] [--temporary] DET_ID
    km3pipe runinfo [--temporary] DET_ID RUN
    km3pipe rundetsn [--temporary] RUN DETECTOR
    km3pipe retrieve DET_ID RUN [-o OUTFILE]
    km3pipe (-h | --help)
    km3pipe --version

Options:
    -h --help           Show this screen.
    -m                  Get the MC detector file (flips the sign of DET_ID).
    -c CALIBR_ID        Geometrical calibration ID (eg. A01466417)
    -o OUTFILE          Output filename.
    -t T0_SET           Time calibration ID (eg. A01466431)
    -n EVENTS/RUNS      Number of events/runs.
    -s REGEX            Regular expression to filter the runsetup name/id.
    --temporary         Use a temporary session. [default: False]
    --overwrite         Overwrite existing config [default: False]
    DET_ID              Detector ID (eg. D_ARCA001).
    DETECTOR            Detector (eg. ARCA).
    GIT_BRANCH          Git branch to pull (eg. develop).
    RUN                 Run number.
"""

from __future__ import division, absolute_import, print_function

import sys
import os
from datetime import datetime

from . import version
from .tools import irods_filepath
from .db import DBManager
from .hardware import Detector

from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def run_tests():
    import pytest
    import km3pipe
    pytest.main([os.path.dirname(km3pipe.__file__)])


def runtable(det_id, n=5, sep='\t', regex=None, temporary=False):
    """Print the run table of the last `n` runs for given detector"""
    db = DBManager(temporary=temporary)
    df = db.run_table(det_id)

    if regex is not None:
        df = df[df['RUNSETUPNAME'].str.match(regex) |
                df['RUNSETUPID'].str.match(regex)]

    if n is not None:
        df = df.tail(n)

    df.to_csv(sys.stdout, sep=sep)


def runinfo(run_id, det_id, temporary=False):
    db = DBManager(temporary=temporary)
    df = db.run_table(det_id)
    row = df[df['RUN'] == run_id]
    if len(row) == 0:
        log.error("No database entry for run {0} found.".format(run_id))
        return
    next_row = df[df['RUN'] == (run_id + 1)]
    if len(next_row) != 0:
        end_time = next_row['DATETIME'].values[0]
        duration = (next_row['UNIXSTARTTIME'].values[0] -
                    row['UNIXSTARTTIME'].values[0]) / 1000 / 60
    else:
        end_time = duration = float('NaN')
    print("Run {0} - detector ID: {1}".format(run_id, det_id))
    print('-' * 42)
    print("  Start time:         {0}\n"
          "  End time:           {1}\n"
          "  Duration [min]:     {2:.2f}\n"
          "  Start time defined: {3}\n"
          "  Runsetup ID:        {4}\n"
          "  Runsetup name:      {5}\n"
          "  T0 Calibration ID:  {6}\n"
          .format(row['DATETIME'].values[0],
                  end_time,
                  duration,
                  bool(row['STARTTIME_DEFINED'].values[0]),
                  row['RUNSETUPID'].values[0],
                  row['RUNSETUPNAME'].values[0],
                  row['T0_CALIBSETID'].values[0]))


def update_km3pipe(git_branch=''):
    if git_branch == '' or git_branch is None:
        git_branch = 'master'
    os.system("pip install -U git+http://git.km3net.de/km3py/km3pipe.git@{0}"
              .format(git_branch))


def retrieve(run_id, det_id, outfile=None):
    """Retrieve run from iRODS for a given detector (O)ID"""
    try:
        det_id = int(det_id)
    except ValueError:
        pass
    path = irods_filepath(det_id, run_id)
    suffix = '' if outfile is None else outfile
    os.system("iget -Pv {0} {1}".format(path, suffix))


def detx(det_id, calibration='', t0set='', filename=None):
    now = datetime.now()
    if filename is None:
        filename = "KM3NeT_{0}{1:08d}_{2}.detx" \
                   .format('-' if det_id < 0 else '', abs(det_id),
                           now.strftime("%d%m%Y"))
    det = Detector(det_id=det_id, t0set=t0set, calibration=calibration)
    if det.n_doms > 0:
        det.write(filename)


def detectors(regex=None, sep='\t', temporary=False):
    """Print the detectors table"""
    db = DBManager(temporary=temporary)
    dt = db.detectors
    if regex is not None:
        dt = dt[dt['OID'].str.match(regex) | dt['CITY'].str.match(regex)]
    dt.to_csv(sys.stdout, sep=sep)


def rundetsn(run_id, detector="ARCA", temporary=False):
    """Print the detector serial number for a given run of ARCA/ORCA"""
    db = DBManager(temporary=temporary)
    dts = db.detectors
    for det_id in dts[dts.OID.str.contains(detector)].SERIALNUMBER:
        if run_id in db.run_table(det_id).RUN.values:
            print(det_id)
            return


def createconf(overwrite=False, dump=False):
    import os.path
    from os import environ
    defaultconf = '[General]\ncheck_for_updates=yes'
    if dump:
        print("--------------\n" + defaultconf + "\n--------------")
    fname = environ['HOME'] + '/.km3net'
    if not overwrite and os.path.exists(fname):
        log.warn('Config exists, not overwriting')
        return
    with open(fname, 'w') as f:
        f.write(defaultconf)
    print('Wrote default config to ', fname)
    os.chmod(fname, 0o600)


def main():
    from docopt import docopt
    args = docopt(__doc__, version=version)

    try:
        n = int(args['-n'])
    except TypeError:
        n = None

    if args['test']:
        run_tests()

    if args['update']:
        update_km3pipe(args['GIT_BRANCH'])

    if args['createconf']:
        overwrite_conf = bool(args['--overwrite'])
        dump = bool(args['--dump'])
        createconf(overwrite_conf, dump)

    if args['runtable']:
        runtable(args['DET_ID'], n, regex=args['-s'],
                 temporary=args["--temporary"])

    if args['runinfo']:
        runinfo(int(args['RUN']), args['DET_ID'],
                temporary=args["--temporary"])

    if args['rundetsn']:
        rundetsn(int(args['RUN']), args['DETECTOR'],
                 temporary=args["--temporary"])

    if args['retrieve']:
        retrieve(int(args['RUN']), args['DET_ID'], args['-o'])

    if args['detx']:
        t0set = args['-t']
        calibration = args['-c']
        outfile = args['-o']
        det_id = int(('-' if args['-m'] else '') + args['DET_ID'])
        detx(det_id, calibration, t0set, outfile)

    if args['detectors']:
        detectors(regex=args['-s'], temporary=args["--temporary"])
