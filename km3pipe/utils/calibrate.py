#!/usr/bin/env python
# coding=utf-8
# Filename: calibrate.py
"""
Apply geometry and time calibration from a DETX to an HDF5 file.

Usage:
    calibrate DETXFILE HDF5FILE
    calibrate (-h | --help)
    calibrate --version

Options:
    -h --help       Show this screen.
"""
from __future__ import division

import km3pipe as kp

import numpy as np
import tables as tb


FILTERS = tb.Filters(complevel=5, shuffle=True, fletcher32=True, complib='zlib')


def calibrate_hits(f, geo):
    dom_ids = f.get_node("/hits/dom_id")[:]
    channel_ids = f.get_node("/hits/channel_id")[:]
    n = len(dom_ids)
    calib = np.empty((n, 7))
    for i in range(n):
        dom_id = dom_ids[i]
        channel_id = channel_ids[i]
        calib[i] = geo._pos_dir_t0_by_dom_and_channel[dom_id][channel_id]
    apply_calibration(calib, f, n, "/hits")


def calibrate_mc_hits(f, geo):
    pmt_ids = f.get_node("/mc_hits/pmt_id")[:]
    n = len(pmt_ids)
    calib = np.empty((n, 7))
    for i in range(n):
        pmt_id = pmt_ids[i]
        calib[i] = geo._pos_dir_t0_by_pmt_id[pmt_id]
    apply_calibration(calib, f, n, "/mc_hits")


def apply_calibration(calib, f, n, loc):
    f8_atom = tb.Float64Atom()
    for i, node in enumerate([p+'_'+s for p in ['pos', 'dir'] for s in 'xyz']):
        print("  ...creating " + node)
        ca = f.create_carray(loc, node, f8_atom, (n,), filters=FILTERS)
        ca[:] = calib[:, i]
    if loc == "/hits":
        print("  ...creating t0")
        print("  ...adding t0s to hit times")
        ca = f.create_carray(loc, "t0", f8_atom, (n,), filters=FILTERS)
        ca[:] = calib[:, 6]
        f.get_node(loc + "/time")[:] += calib[:, 6]


def main():
    from docopt import docopt

    args = docopt(__doc__)

    with tb.File(args['HDF5FILE'], "a") as f:
        try:
            kp.io.hdf5.check_version(f, args['HDF5FILE'])
        except kp.io.hdf5.H5VersionError as e:
            print(e)
            raise SystemExit

        if("/hits/pos_x" in f or "/mc_hits/pos_y" in f):
            print("The file seems to be calibrated, please check.")
            raise SystemExit

        print("Reading geometry information")
        geo = kp.Geometry(filename=args['DETXFILE'])

        if '/hits' in f:
            print("Calibrating hits")
            calibrate_hits(f, geo)
        if '/mc_hits' in f:
            print("Calibrate MC hits")
            calibrate_mc_hits(f, geo)

        print("Done.")


if __name__ == "__main__":
    main()
