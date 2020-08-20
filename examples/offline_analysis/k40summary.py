#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: ts=4 sw=4 et
"""
=======================
K40 Calibration Summary
=======================

Combine k40calib results into a single CSV file.

.. code-block:: bash

    Usage:
        k40summary.py CALIB_FOLDER
        k40summary.py (-h | --help)

    Options:
        -h --help   Show this screen.

"""

from glob import glob
import os
import re
import pickle
import numpy as np


def write_header(fobj):
    """Add the header to the CSV file"""
    fobj.write("# K40 calibration results\n")
    fobj.write("det_id\trun_id\tdom_id")
    for param in ["t0", "qe"]:
        for i in range(31):
            fobj.write("\t{}_ch{}".format(param, i))


def main():
    from docopt import docopt

    args = docopt(__doc__)

    file_pattern = os.path.join(args["CALIB_FOLDER"], "*.p")
    files = glob(file_pattern)

    with open("k40calib_summary.csv", "w") as csv_file:
        write_header(csv_file)

        for fn in files:
            det_id, run_id = [int(x) for x in re.search("_(\\d{8})" * 2, fn).groups()]
            with open(fn, "rb") as f:
                data = pickle.load(f)
            if not data:
                print("Empty dataset found for '{}'".format(fn))
            else:
                for dom_id in data.keys():
                    t0s = data[dom_id]["opt_t0s"].x
                    qes = data[dom_id]["opt_qes"].x
                    cols = np.concatenate([[det_id, run_id, dom_id], t0s, qes])
                    line = "\n" + "\t".join(str(c) for c in cols)
                    csv_file.write(line)


if __name__ == "__main__":
    main()
