# coding=utf-8
"""
Print the structure of a hdf5 file.

This is a less verbose version of ``ptdump`` from the pytables package.
If you want much more detailed (+verbose) output, e.g. for debugging, by
all means use the ``ptdump`` util (it's already installed as a dependency
alongside km3pipe).

Usage:
    h5tree [options] FILE
    h5tree (-h | --help)
    h5tree --version

Options:
    FILE        Input file.
    -h --help   Show this screen.
    --no-meta   Don't print meta data at top
    --titles    Print leaf titles.

"""
from __future__ import division, absolute_import, print_function

import numpy as np
import tables

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2016, Moritz Lotze and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"


def nodeinfo(node, print_titles=False):
    pathname = node._v_pathname
    try:
        n_rows = node.shape[0]
        n_rows_str = '{}'.format(n_rows)
    except AttributeError:
        n_rows_str = None
    try:
        title = str(node.title)
        if not print_titles or not title or title.isspace():
            title = None
        else:
            title = "'{}'".format(title)
    except AttributeError:
        title = None
    return ", ".join([node for node in (pathname, title, n_rows_str)
                      if node is not None])


def meta(h5):
    try:
        version = np.string_(h5.root._v_attrs.format_version)
        print('KM3HDF5 v{}'.format(version.decode('utf-8')))
    except AttributeError:
        pass
    try:
        info = h5.root.event_info
        n_events = info.shape[0]
        print("Number of Events: {}".format(n_events))
    except tables.NoSuchNodeError:
        pass


def h5tree(h5name, print_meta=True, **kwargs):
    with tables.open_file(h5name) as h5:
        if print_meta:
            meta(h5)
        node_kinds = h5.root._v_file._node_kinds[1:]
        what = h5.root._f_walk_groups()
        for group in what:
            print(str(group))
            for kind in node_kinds:
                for node in group._f_list_nodes(kind):
                    print(nodeinfo(node, **kwargs))


def main():
    from docopt import docopt
    args = docopt(__doc__)
    fname = args['FILE']
    do_titles = bool(args['--titles'])
    do_meta = not bool(args['--no-meta'])
    h5tree(fname,
           print_titles=do_titles,
           print_meta=do_meta,
          )
