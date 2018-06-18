# Filename: cmd.py
"""
Convert HDF5 to vanilla ROOT.

Can convert multiple files at once, "foo.h5" -> "foo.h5.root".

Usage:
     hdf2root [--verbose] FILES...
     hdf2root (-h | --help)
     hdf2root --version

Options:
    -h --help           Show this screen.
    --verbose           Print more info [default: False]
"""

import numpy as np

from km3pipe import version

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def hdf2root(infile, outfile, verbose=False):
    try:
        from rootpy.io import root_open
        from rootpy import asrootpy
        from root_numpy import array2tree
    except ImportError:
        raise ImportError(
            "Please load ROOT into PYTHONPATH and install rootpy+root_numpy:\n"
            "   `pip install rootpy root_numpy`"
        )

    from tables import open_file

    h5 = open_file(infile, 'r')
    rf = root_open(outfile, 'recreate')

    # 'walk_nodes' does not allow to check if is a group or leaf
    #   exception handling is bugged
    #   introspection/typecheck is buged
    # => this moronic nested loop instead of simple `walk`
    for group in h5.walk_groups():
        for leafname, leaf in group._v_leaves.items():
            arr = leaf[:]
            if arr.dtype.names is None:
                dt = np.dtype((arr.dtype, [(leafname, arr.dtype)]))
                arr = arr.view(dt)
            treename = leaf._v_pathname.replace('/', '_')
            tree = asrootpy(array2tree(arr, name=treename))
            tree.write()
    rf.close()
    h5.close()


def main():
    from docopt import docopt
    args = docopt(__doc__, version=version)

    files = args['FILES']
    verbose = bool(args['--verbose'])
    for fn in files:
        print('Converting {}...'.format(fn))
        hdf2root(fn, fn + '.root', verbose=verbose)
