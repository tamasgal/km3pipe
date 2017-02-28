#!/usr/bin/env python
# coding=utf-8
# Filename: ptconcat.py
"""
Concatenate HDF5 Files.

Usage:
    ptconcat [--verbose] [--ignore-id] OUTFILE INFILES...
    ptconcat (-h | --help)
    ptconcat --version

Options:
    -h --help               Show this screen.
    --verbose               Print out more progress. [default: False].
"""

from __future__ import division, absolute_import, print_function
import os.path
from six import iteritems

import tables as tb

from km3pipe import version

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"


def ptconcat(output_file, input_files, verbose=False):
    """Concatenate HDF5 Files"""
    filt = tb.Filters(complevel=5, shuffle=True,
                      fletcher32=True, complib='zlib')
    out_tabs = {}
    h5struc = tb.open_file(input_files[0])
    h5out = tb.open_file(output_file, 'a')
    for node in h5struc.walk_nodes('/', classname='Table'):
        path = node._v_pathname
        dtype = node.dtype
        p, n = os.path.split(path)
        out_tabs[path] = h5out.create_table(p, n, description=dtype,
                                            filters=filt)
    h5struc.close()
    for fname in input_files:
        h5 = tb.open_file(fname)
        for path, out in iteritems(out_tabs):
            tab = h5.get_node(path)
            out.append(tab[:])
        h5.close()
    h5out.close()


def main():
    from docopt import docopt
    args = docopt(__doc__, version=version)

    infiles = args['INFILES']
    outfile = args['OUTFILE']
    verb = bool(args['--verbose'])

    ptconcat(outfile, infiles, verbose=verb)


if __name__ == '__main__':
    main()
