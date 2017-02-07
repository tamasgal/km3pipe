# coding=utf-8
# Filename: k40calib.py
"""
Calculate t0set and write it to a CSV file.

Usage:
    k40calib FILE [-o OUTFILE]
    k40calib (-h | --help)
    k40calib --version

Options:
    -o OUTFILE   CSV file containing the t0 values.
    -h --help    Show this screen.
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


def k40calib(input_file, output_file=None):
    """K40 Calibration"""
    from km3modules import k40
    import km3pipe as kp
    import ROOT

    f = ROOT.TFile(input_file)
    dom_ids = [n.GetName().split('.')[0]
               for n in f.GetListOfKeys()
               if '.2S' in n.GetName()]
    detector = kp.hardware.Detector(det_id=14)
    if output_file is None:
        csv_filename = input_file + "_t0s.csv"
    else:
        csv_filename = output_file
    csv_file = open(csv_filename, 'w')
    csv_file.write("dom_id,tdc_channel,t0\n")
    for dom_id in dom_ids:
        print("Calibrating {0}...".format(dom_id))
        try:
            calibration = k40.calibrate_dom(dom_id, input_file, detector)
        except ValueError:
            print("   no data found, skipping.")
            continue
        t0set = calibration['opt_t0s'].x
        for tdc_channel, t0 in enumerate(t0set):
            csv_file.write("{0}, {1}, {2}\n"
                           .format(dom_id, tdc_channel, t0))
        print("    done.")
    csv_file.close()
    print("Calibration done, the t0 values were written to '{0}'."
          .format(csv_filename))


def main():
    from docopt import docopt
    args = docopt(__doc__, version=version)

    infile = args['FILE']
    k40calib(infile, args['-o'])
