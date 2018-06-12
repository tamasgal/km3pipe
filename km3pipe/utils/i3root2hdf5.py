#!/usr/bin/env python
"""Robust root->h5 converter, for rootfiles generated from I3 files.

Certain keys *have* to be skipped, otherwise the converter segfaults (!!!).

Usage:
    i3root2hdf5.py [--force] INFILE
    i3root2hdf5.py -h | --help


Options:
    -h --help     Show this screen.
    --force         Overwrite target if it already exists.
"""

from docopt import docopt

import h5py
from rootpy.io import root_open
from root_numpy import tree2array


def i3root2hdf5(infile, force=False):
    h5file = infile + '.h5'
    bad_keys = ['AntMCTree', 'MasterTree']
    rf = root_open(infile, 'r')
    keys = [k.name for k in rf.keys()]
    if force:
        mode = 'w'
    else:
        mode = 'a'
    h5 = h5py.File(h5file, mode)
    for key in keys:
        if key in bad_keys:
            continue
        tree = rf[key]
        arr = tree2array(tree)
        try:
            h5.create_dataset(
                key,
                data=arr,
                compression='gzip',
                compression_opts=9,
                shuffle=True,
                fletcher32=True,
            )
        except TypeError:
            continue
        h5.flush()
    h5.close()


def main():
    args = docopt(__doc__)
    infile = args['INFILE']
    force = args['--force']
    i3root2hdf5(infile, force=force)


if __name__ == '__main__':
    main()
