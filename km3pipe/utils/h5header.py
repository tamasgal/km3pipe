# Filename: h5header.py
"""
Show the km3pipe etc. version used to write a H5 file.

Usage:
    h5header FILE [-r]
    h5header (-h | --help)
    h5header --version

Options:
    FILE        Input file.
    -r --raw    Dump raw metadata.
    -h --help   Show this screen.

"""

import tables

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__credits__ = "Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def h5header(h5name, raw=False):
    with tables.open_file(h5name) as h5:
        header_loc = "/header"
        if not header_loc in h5:  # noqa
            print("Sorry, no header.")
            return
        header = h5.get_node(header_loc)
        att_list = header._v_attrs._f_list()
        if raw:
            print(header._v_attrs.__repr__())
            return
        if not att_list and not raw:
            print("Sorry, no metadata.")
            return
        for att in header._v_attrs._f_list():
            ver = header._v_attrs[att]
            print("{}: {}".format(att, ver))


def main():
    from docopt import docopt

    arguments = docopt(__doc__)

    h5header(arguments["FILE"], arguments["--raw"])
