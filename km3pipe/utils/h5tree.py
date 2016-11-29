# coding=utf-8
# Filename: h5tree.py
"""
Print the HDF5 file structure.

Usage:
    h5tree FILE
    h5tree (-h | --help)
    h5tree --version

Options:
    FILE       Input file.
    -h --help  Show this screen.

"""
from __future__ import division, absolute_import, print_function

import tables

from km3pipe.tools import deprecated

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2016, Moritz Lotze and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"


MSG = "Use `ptdump` from pytables instead, it's better!"

@deprecated(MSG)
def h5tree(h5name):
    with tables.open_file(h5name) as h5:
        for node in h5.walk_nodes():
            print(node)


def main():
    from docopt import docopt
    arguments = docopt(__doc__)

    print('DeprecationWarning: h5tree.py is deprecated. ' + MSG)
    h5tree(arguments['FILE'])
