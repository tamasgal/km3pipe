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
            print("HDF5 Meta Data")
            print("--------------")
            print(h5.root._v_attrs.__repr__())
            if '/header' in h5:
                print("\nHeader ('/header')")
                print("----------------")
                print(h5.get_node('/header')._v_attrs.__repr__())
            return
        if not att_list and not raw:
            print("Sorry, no metadata.")
        else:
            print("HDF5 Meta Data")
            print("--------------")
        for att in h5.root._v_attrs._f_list():
            ver = h5.root._v_attrs[att]
            print("{}: {}".format(att, ver))
        if '/header' in h5:
            print("\nHeader (/header)")
            print("----------------")
            node = h5.get_node('/header')
            for att in node._v_attrs._f_list():
                ver = node._v_attrs[att]
                print("{}: {}".format(att, ver))


def main():
    from docopt import docopt
    arguments = docopt(__doc__)

    h5info(arguments['FILE'], arguments['--raw'])
