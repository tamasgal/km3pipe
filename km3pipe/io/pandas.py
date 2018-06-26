# Filename: pandas.py
# pylint: disable=C0103,R0903
# vim:set ts=4 sts=4 sw=4 et:
"""
Pandas Helpers.
"""
from __future__ import absolute_import, print_function, division

import os.path

import numpy as np
import pandas as pd
import tables as tb

from km3pipe.logger import get_logger
from km3pipe.tools import insert_prefix_to_dtype

log = get_logger(__name__)    # pylint: disable=C0103

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

    Examples
    --------

    >>> filenames = ['numu_cc.h5', 'anue_nc.h5']
    >>> c = H5Chain(filenames)
    >>> X = c['/reco/gandalf']

    A context manager is also available:

    >>> with H5Chain(filenames) as h5:
    >>>     reco = h5['/reco/foo']

    """

    def __init__(self, filenames):
        self.filenames = filenames
        self.log = get_logger(self.__class__.__name__)

    def close(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __enter__(self):
        return self

    def __getitem__(self, key):
        dfs = []
        for fname in self.filenames:
            self.log.info('opening ', fname)
            with tb.File(fname, 'r') as h5:
                try:
                    tab = h5.get_node(key)[:]
                except KeyError as ke:
                    self.log.error(
                        '{} does not exist in {}!'.format(key, fname)
                    )
                    raise ke
                except tb.exceptions.NodeError as ne:
                    self.log.error(
                        '{} does not exist in {}!'.format(key, fname)
                    )
                    raise ne
            self.log.debug("Table shape: {}".format(tab.shape))
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


def drop_duplicate_columns(df):
    """If a column name appears more than once, drop it."""
    # _, i = np.unique(df.columns, return_index=True)
    # return df.iloc[:, i]
    return df.T.drop_duplicates().T


def merge_event_ids(df, drop_duplicates=False):
    cols = list(df.columns)
    if drop_duplicates:
        # that function some times reaches max recursion depth, whyever
        cols = drop_duplicate_columns(df)
    ids = list(c for c in cols if 'event_id' in c)
    log.debug(ids)
    if not ids:
        return df
    ids = list(set(ids))
    log.debug(ids)
    # non_id = list(c for c in cols if c not in ids)
    event_ids = df[ids[0]]
    try:
        log.debug(event_ids.shape)
        if event_ids.shape[1] > 1:
            event_ids = event_ids.ix[:, 0]
        log.debug(event_ids.shape)
    except IndexError:
        pass
    df.drop(ids, axis=1, inplace=True)
    log.debug(event_ids.shape)
    df['event_id'] = event_ids
    return df


def df_to_h5(df, h5file, where, **kwargs):
    """Write pandas dataframes with proper columns.

    Example:
        >>> df = pd.DataFrame(...)
        >>> df_to_h5(df, 'foo.h5', '/some/loc/my_df')
    """
    write_table(df.to_records(index=False), h5file, where, **kwargs)


def write_table(
        array,
        h5file,
        where,
        force=False,
        filters=None,
        createparents=True,
        **kwargs
):
    """Write a structured numpy array into a H5 table.
    """
    own_h5 = False
    if isinstance(h5file, str):
        own_h5 = True
        h5file = tb.open_file(h5file, 'a')
    if filters is None:
        filters = tb.Filters(complevel=5, shuffle=True, fletcher32=True)
    loc, tabname = os.path.split(where)
    if loc == '':
        loc = '/'
    if force:
        try:
            h5file.remove_node(loc, tabname, recursive=True)
        except tb.NoSuchNodeError:
            log.warn(
                'Force -> Trying to remove+rewrite of table at {}, {}, '
                'but it did not previously exists.'.format(loc, tabname)
            )
    h5file.create_table(
        loc,
        tabname,
        obj=array,
        createparents=createparents,
        filters=filters,
        **kwargs
    )
    if own_h5:
        h5file.close()
