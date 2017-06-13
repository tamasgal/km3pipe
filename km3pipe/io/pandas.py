# coding=utf-8
# Filename: pandas.py
# pylint: disable=C0103,R0903
# vim:set ts=4 sts=4 sw=4 et:
"""
Pandas Helpers.
"""
from __future__ import division, absolute_import, print_function

import os.path
from six import string_types

import numpy as np
import pandas as pd
import tables as tb

from km3pipe.logger import logging
from km3pipe.tools import insert_prefix_to_dtype

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"


class H5Chain(object):
    """Read/write multiple HDF5 files as ``pandas.DataFrame``.

    It is impliend that all files share the same group/tables
    structure and naming.

    Parameters
    ----------
    filenames: list(str), or dict(fname -> h5file)
    verbose: bool [default: False]

    Examples
    --------

    >>> filenames = ['numu_cc.h5', 'anue_nc.h5']
    >>> c = H5Chain(filenames)
    >>> X = c['/reco/gandalf']

    A context manager is also available:

    >>> with H5Chain(filenames) as h5:
    >>>     reco = h5['/reco']

    """
    def __init__(self, filenames, verbose=False):
        self.filenames = filenames
        self.verbose = verbose

    def close(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __enter__(self):
        return self

    def __getitem__(self, key):
        dfs = []
        for fname in self.filenames:
            if self.verbose:
                print('opening ', fname)
            with tb.File(fname, 'r') as h5:
                try:
                    tab = h5.get_node(key)[:]
                except KeyError as ke:
                    log.error('{} does not exist in {}!'.format(key, fname))
                    raise ke
                except tb.exceptions.NodeError as ne:
                    log.error('{} does not exist in {}!'.format(key, fname))
                    raise ne
            if self.verbose:
                print(tab.shape)
            df = pd.DataFrame(tab)
            dfs.append(df)
        dfs = pd.concat(dfs, axis=0, ignore_index=True)
        return dfs

    def __call__(self, key):
        """
        Parameters
        ----------
        key: str
            H5 path of the object to retrieve, e.g. '/reco/gandalf'.
        """
        return self[key]


def map2df(map):
    return pd.DataFrame.from_records(map, index=np.ones(1, dtype=int))


def _read_table(tab, max_id=None, ignore_events=False):
    if ignore_events:
        return tab[:max_id]
    else:
        return tab.read_where('event_id <= %d' % max_id)


def read_group(group, max_id=None, **kwargs):
    # Store through groupname, insert tablename into dtype
    df = []
    for tab in group._f_iter_nodes(classname='Table'):
        tabname = tab.name
        if max_id is None:
            arr = tab.read(**kwargs)
        else:
            arr = _read_table(tab, max_id)
        arr = insert_prefix_to_dtype(arr, tabname)
        arr = pd.DataFrame.from_records(arr)
        df.append(arr)
    df = pd.concat(df, axis=1)
    return df


def merge_event_ids(df):
    cols = list(df.columns)
    ids = list(c for c in cols if 'event_id' in c)
    if not ids:
        return df
    # non_id = list(c for c in cols if c not in ids)
    event_ids = df[ids[0]]
    df.drop(ids, axis=1, inplace=True)
    df['event_id'] = event_ids
    return df


def df_to_h5(df, h5file, where, **kwargs):
    """Write pandas dataframes with proper columns.

    Example:
        >>> df = pd.DataFrame(...)
        >>> df_to_h5(df, 'foo.h5', '/some/loc/my_df')
    """
    write_table(df.to_records(index=False), h5file, where, **kwargs)


def write_table(array, h5file, where, force=False):
    """Write a structured numpy array into a H5 table.
    """
    own_h5 = False
    if isinstance(h5file, string_types):
        own_h5 = True
        h5file = tb.open_file(h5file, 'a')
    filt = tb.Filters(complevel=5, shuffle=True, fletcher32=True)
    loc, tabname = os.path.split(where)
    if loc == '':
        loc = '/'
    try:
        h5file.create_table(loc, tabname, obj=array, createparents=True,
                            filters=filt)
    except tb.exceptions.NodeError:
        h5file.get_node(where)[:] = array
    if own_h5:
        h5file.close()


def first_mc_tracks(mc_tracks, mupage=False):
    mc_tracks = pd.DataFrame(mc_tracks)
    if mupage:
        mc_tracks = mc_tracks[mc_tracks.type != 0]
        mc_tracks = mc_tracks[mc_tracks.id == 1]
    return mc_tracks.drop_duplicates(subset='event_id')
