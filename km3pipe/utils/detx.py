# Filename: detx.py
"""
Retrieves the DETX with the latest t0 calibration for a given detector ID
and run. If only the detector ID is provided, the most recent t0 calibration
is used.

Usage:
    runtable [options] DET_ID [RUN_ID]
    runtable (-h | --help)
    runtable --version

Options:
    -h --help           Show this screen.
    DET_ID              Detector ID (eg. 42).
    RUN_ID              Run ID (eg. 4012).

"""

import re
import sys
import km3pipe as kp

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2019, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = kp.logger.get_logger(__name__)


def detx(det_id, run_id):
    now = datetime.now()
    if filename is None:
        filename = "KM3NeT_{0}{1:08d}_{2}{3}{4}.detx" \
                   .format('-' if det_id < 0 else '',
                           abs(det_id),
                           now.strftime("%d%m%Y"),
                           '_t0set-%s' % t0set if t0set else '',
                           '_calib-%s' % calibration if calibration else '',
                           )
    det = Detector(det_id=det_id, t0set=t0set, calibration=calibration)
    if det.n_doms > 0:
        det.write(filename)


def main():
    from docopt import docopt
    args = docopt(__doc__, version=kp.version)

    try:
        n = int(args['-n'])
    except TypeError:
        n = None

    detx(
        args['DET_ID'],
        args['RUN_ID'],
    )
