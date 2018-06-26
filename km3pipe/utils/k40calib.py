#!/usr/bin/env python
# Filename: k40calib.py
# vim: ts=4 sw=4 et
"""
=========================
K40 Intra-DOM Calibration
=========================

The following script calculates the PMT time offsets using K40 coincidences


Usage:
    k40calib FILE DET_ID [-t TMAX -c CTMIN -r -o CALIB_FILE -s STREAM]
    k40calib (-h | --help)
    k40calib --version

Options:
    FILE            Input file (ROOT).
    DET_ID          Detector ID (e.g. 29).
    -r              Skip frames with with at least one PMT in HRV.
    -t TMAX         Coincidence time window [default: 10].
    -s STREAM       JDAQTimeslice stream (L1, L2, SN, ...) [default: ].
    -c CTMIN        Minimum cos(angle) between PMTs for L2 [default: -1.0].
    -o CALIB_FILE   Filename for the calibration output [default: k40_cal.p].
    -h --help       Show this screen.
"""
# Author: Jonas Reubelt <jreubelt@km3net.de> and Tamas Gal <tgal@km3net.de>
# License: MIT

from km3pipe import version
import km3pipe as kp
from km3modules import k40
from km3modules.common import StatusBar, MemoryObserver

__author__ = "Tamas Gal and Jonas Reubelt"
__copyright__ = "Copyright 2016, KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Jonas Reubelt and Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def k40calib(
        filename, tmax, ctmin, stream, filter_hrv, det_id, calib_filename
):
    pipe = kp.Pipeline()
    pipe.attach(kp.io.jpp.TimeslicePump, filename=filename, stream=stream)
    pipe.attach(StatusBar, every=5000)
    pipe.attach(MemoryObserver, every=10000)
    pipe.attach(
        k40.HRVFIFOTimesliceFilter, filter_hrv=filter_hrv, filename=filename
    )
    pipe.attach(k40.SummaryMedianPMTRateService, filename=filename)
    pipe.attach(k40.TwofoldCounter, tmax=tmax)
    pipe.attach(k40.K40BackgroundSubtractor, mode='offline')
    pipe.attach(
        k40.IntraDOMCalibrator,
        ctmin=ctmin,
        mode='offline',
        det_id=det_id,
        calib_filename=calib_filename
    )
    pipe.drain()


def main():
    from docopt import docopt
    args = docopt(__doc__, version=version)

    k40calib(
        args['FILE'], int(args['-t']), float(args['-c']), args['-s'],
        args['-r'], int(args['DET_ID']), args['-o']
    )
