# coding=utf-8
# Filename: k40calib.py
"""
Convert ROOT and EVT files to HDF5.

Usage:
    k40calib FILE
    k40calib (-h | --help)
    k40calib --version

Options:
    -h --help  Show this screen.
"""

from __future__ import division, absolute_import, print_function

from km3pipe import version

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Jonas Reubelt and Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def k40calib(input_file):
    """K40 Calibration"""
    from km3modules import k40
    import km3pipe as kp
    import ROOT
    f = ROOT.TFile(input_file)
    dom_ids = [n.GetName().split('.')[0]
               for n in f.GetListOfKeys()
               if '.2S' in n.GetName()]
    detector = kp.hardware.Detector(det_id=14)
    for dom_id in dom_ids:
        calibration = k40.calibrate_dom(dom_id, input_file, detector)
        print(calibration.keys())

def main():
    from docopt import docopt
    args = docopt(__doc__, version=version)

    infile = args['FILE']
    k40calib(infile)
