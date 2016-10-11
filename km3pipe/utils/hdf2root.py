# coding=utf-8
# Filename: cmd.py
"""
Convert HDF5 to vanilla ROOT.

Usage:
     hdf2root FILE [-o OUTFILE]
     hdf2root (-h | --help)
     hdf2root --version

Options:
    -h --help           Show this screen.
    -o OUTFILE          Output file.
"""

from __future__ import division, absolute_import, print_function

import sys
import os
from datetime import datetime

from km3pipe import version
from km3modules import StatusBar

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def hdf2root(infile, outfile):
    from rootpy.io import root_open
    from rootpy import asrootpy
    from root_numpy import array2tree
    from tables import open_file

    h5 = open_file(infile, 'r')
    rf = root_open(outfile, 'recreate')

    # 'walk_nodes' does not allow to check if is a group or leaf
    #   exception handling is bugged
    #   introspection/typecheck is buged
    # => this moronic nested loop instead of simple `walk`
    for group in h5.walk_groups():
        for leafname, leaf in group._v_leaves.items():
            tree = asrootpy(array2tree(leaf[:], name=leaf._v_pathname))
            tree.write()
    rf.close()
    h5.close()


def main():
    from docopt import docopt
    args = docopt(__doc__, version=version)

    infile = args['FILE']
    outfile = args['-o'] or infile + '.root'
    hdf2root(infile, outfile)
