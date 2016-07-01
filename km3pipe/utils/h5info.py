# coding=utf-8
# Filename: h5tree.py
"""
Show the km3pipe etc. version used to write a H5 file.

Usage:
    h5info FILE [-r]
    h5info (-h | --help)
    h5info --version

Options:
    FILE        Input file.
    -r --raw    Dump raw metadata.
    -h --help   Show this screen.

"""
from __future__ import division, absolute_import, print_function

import tables

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2016, Moritz Lotze and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"


def h5info(h5name, raw=False):
    with tables.open_file(h5name) as h5:
        att_list = h5.root._v_attrs._f_list()
        if raw:
            print(h5.root._v_attrs.__repr__())
            return
        if not att_list and not raw:
            print("Sorry, no metadata.")
        for att in h5.root._v_attrs._f_list():
            ver = h5.root._v_attrs[att]
            print("{}: {}".format(att, ver))


def main():
    from docopt import docopt
    arguments = docopt(__doc__)

    h5info(arguments['FILE'], arguments['--raw'])
