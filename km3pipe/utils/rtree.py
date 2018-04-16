"""
Print the ROOT file structure.

Usage:
    rtree [options] FILE
    rtree (-h | --help)
    rtree --version

Options:
    FILE       Input file.
    -h --help  Show this screen.

"""
from km3pipe.io.root import open_rfile

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2016, Moritz Lotze and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"


def rtree(rfile):
    rfile = open_rfile(rfile)
    print(rfile.ls())
    rfile.close()


def main():
    from docopt import docopt
    arguments = docopt(__doc__)

    rtree(arguments['FILE'])
