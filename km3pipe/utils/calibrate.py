#!/usr/bin/env python
# Filename: calibrate.py
"""
Apply geometry and time calibration from a DETX to an HDF5 file.

Usage:
    calibrate [-c CHUNK_SIZE] DETXFILE HDF5FILE
    calibrate (-h | --help)
    calibrate --version

Options:
    -c CHUNK_SIZE   Size of the chunk to load into memory [default: 200000].
    -h --help       Show this screen.
"""
import os

import km3pipe as kp

import numpy as np
import tables as tb
from tqdm import tqdm

FILTERS = tb.Filters(
    complevel=5, shuffle=True, fletcher32=True, complib='zlib'
)
F4_ATOM = tb.Float32Atom()
U1_ATOM = tb.UInt8Atom()

cprint = kp.logger.get_printer(os.path.basename(__file__))
log = kp.logger.get_logger(os.path.basename(__file__))


def calibrate_hits(f, cal, chunk_size):
    dom_ids = f.get_node("/hits/dom_id")
    channel_ids = f.get_node("/hits/channel_id")
    n_hits = len(dom_ids)
    idx = 0

    chunks = kp.tools.chunks(range(n_hits), chunk_size)
    for chunk in tqdm(chunks, total=(n_hits // chunk_size)):
        n = len(chunk)
        calib = np.empty((n, 9), dtype='f4')

        _dom_ids = dom_ids[idx:idx + n]
        _channel_ids = channel_ids[idx:idx + n]

        idx += n

        for i in range(n):
            dom_id = _dom_ids[i]
            channel_id = _channel_ids[i]
            calib[i] = cal._calib_by_dom_and_channel[dom_id][channel_id]

        write_calibration(calib, f, "/hits")

    f.get_node("/hits")._v_attrs.datatype = "CRawHitSeries"


def calibrate_mc_hits(f, cal):
    pmt_ids = f.get_node("/mc_hits/pmt_id")[:]
    n = len(pmt_ids)
    calib = np.empty((n, 9), dtype='f4')
    for i in range(n):
        pmt_id = pmt_ids[i]
        calib[i] = cal._calib_by_pmt_id[pmt_id]
    write_calibration(calib, f, "/mc_hits")
    f.get_node("/mc_hits")._v_attrs.datatype = "CMcHitSeries"


def write_calibration(calib, f, loc):
    """Write calibration set to file"""
    for i, node in enumerate(
        [p + '_' + s for p in ['pos', 'dir'] for s in 'xyz']):
        h5loc = loc + '/' + node
        ca = f.get_node(h5loc)
        ca.append(calib[:, i])

    du = f.get_node(loc + '/du')
    du.append(calib[:, 7].astype('u1'))

    floor = f.get_node(loc + '/floor')
    floor.append(calib[:, 8].astype('u1'))

    t0 = f.get_node(loc + '/t0')
    t0.append(calib[:, 6])

    if loc == "/hits":
        time = f.get_node(loc + "/time")
        offset = len(time)
        chunk_size = len(calib)
        time[offset - chunk_size:offset] += calib[:, 6]


def initialise_arrays(group, f):
    """Create EArrays for calibrated hits"""
    for node in ['pos_x', 'pos_y', 'pos_z', 'dir_x', 'dir_y', 'dir_z', 'du',
                 'floor', 't0']:
        if node in ['floor', 'du']:
            atom = U1_ATOM
        else:
            atom = F4_ATOM
        f.create_earray(group, node, atom, (0, ), filters=FILTERS)


def main():
    from docopt import docopt

    args = docopt(__doc__)

    with tb.File(args['HDF5FILE'], "a") as f:
        try:
            kp.io.hdf5.check_version(f, args['HDF5FILE'])
        except kp.io.hdf5.H5VersionError as e:
            log.critical(e)
            raise SystemExit

        if ("/hits/pos_x" in f or "/mc_hits/pos_y" in f):
            log.critical("The file seems to be calibrated, please check.")
            raise SystemExit

        cprint("Reading calbration information")
        cal = kp.calib.Calibration(filename=args['DETXFILE'])

        if '/hits' in f:
            initialise_arrays('/hits', f)
            cprint("Calibrating hits")
            calibrate_hits(f, cal, int(args['-c']))
        if '/mc_hits' in f:
            initialise_arrays('/mc_hits', f)
            cprint("Calibrate MC hits")
            calibrate_mc_hits(f, cal, args['CHUNK_SIZE'])

        cprint("Done.")


if __name__ == "__main__":
    main()
