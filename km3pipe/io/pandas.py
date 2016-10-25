# coding=utf-8
# Filename: pandas.py
# pylint: disable=C0103,R0903
# vim:set ts=4 sts=4 sw=4 et:
"""
Pandas Helpers.
"""
from __future__ import division, absolute_import, print_function

from collections import defaultdict, OrderedDict
import os.path

import numpy as np
import pandas as pd
import tables as tb

import km3pipe as kp
from km3pipe import Pump, Module
from km3pipe.dataclasses import ArrayTaco, deserialise_map
from km3pipe.logger import logging
from km3pipe.tools import camelise, decamelise, insert_prefix_to_dtype, split

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

    This class mitigates 3 issues:
    * read different amounts of events from each file
        * (100 from A, 20 from B)
    * don't slice all tables equally, e.g.:
        * read every row from jgandalf
        * read every 2nd row from mc_tracks
    * merge all tables below a group into a single table
        * /reco/jgandalf
        * /reco/dusj

    Parameters
    ----------
    filenames: list(str)

    Examples
    --------
    >>> filenames = ['numu_cc.h5', 'anue_nc.h5']
    >>> n_evts = {'numu_cc.h5': None, 'anue_nc.h5': 100, }
    # either tables keys below '/', or group names
    >>> keys = ['hits', 'reco']
    >>> step = {'mc_tracks': 2}

    >>> c = H5Chain(filenames)
    >>> coll = c(n_evts, keys, step)
    {'mc_tracks': pd.Dataframe, 'hits' pd.DataFrame, 'reco': dataframe}

    >>> # these are pandas Dataframes
    >>> X = coll['reco']
    >>> wgt = coll['event_info']['weights_w2']
    >>> Y_ene = coll['mc_tracks']['energy']
    """

    def __init__(self, filenames):
        self.h5files = {}
        if isinstance(filenames, dict):
            self.h5files.update(filenames)
            return
        for fn in filenames:
            h5 = tb.open_file(fn, 'r')
            self.h5files[fn] = h5

    def close(self):
        for h5 in self.h5files.values():
            h5.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __enter__(self):
        return self

    def __call__(self, n_evts=None, keys=None, ):
        """
        Parameters
        ----------
        n_evts: int or dict(str->int) (default=None)
            Number of events to read. If None, read every event from all.
            If int, read that many from all.  In case of dict: If a
            filename is in the dict, read that many events from the file.

        keys: list(str) (default=None)
            Names of the tables/groups to read. If None, read all.
            Refers only to nodes sitting below '/',
            e.g. '/mc_tracks' (Table) or '/reco' (Group).
        """
        store = defaultdict(list)
        for fname, h5 in self.h5files.items():
            n = n_evts
            if isinstance(n_evts, dict):
                n = n_evts[fname]

            # tables under '/', e.g. mc_tracks
            for tab in h5.iter_nodes('/', classname='Table'):
                tabname = tab.name
                if keys is not None and tabname not in keys:
                    continue
                arr = self._read_table(tab, n)
                arr = pd.DataFrame.from_records(arr)
                store[tabname].append(arr)

            # groups under '/', e.g. '/reco'
            # tables sitting below will be merged
            for gr in h5.iter_nodes('/', classname='Group'):
                groupname = gr._v_name
                if keys is not None and groupname not in keys:
                    continue
                arr = self._read_group(gr, n)
                store[groupname].append(arr)

        for key, dfs in sorted(store.items()):
            store[key] = pd.concat(dfs)
        return store

    @classmethod
    def _read_group(cls, group, n=None, cond=None):
        # Store through groupname, insert tablename into dtype
        df = []
        for tab in group._f_iter_nodes(classname='Table'):
            tabname = tab.name
            arr = cls._read_table(tab, n)
            arr = insert_prefix_to_dtype(arr, tabname)
            arr = pd.DataFrame.from_records(arr)
            df.append(arr)
        df = pd.concat(df, axis=1)
        return df

    @classmethod
    def _read_table(cls, table, n=None, cond=None):
        if isinstance(n, int):
            return table[:n]
        return table[:]

def df_to_h5(df, filename, where, filemode='a', complevel=5,):
    """Write pandas dataframes with proper columns.

    The main 2 ways pandas writes dataframes suck bigly.
    """
    loc, tabname = os.path.split(where)
    if loc == '':
        loc = '/'
    with tb.open_file(filename, filemode) as h5:
        filt = tb.Filters(complevel=complevel, shuffle=True, fletcher32=True)
        h5.create_table(loc, tabname, obj=df.to_records(index=False),
                        filters=filt)


def map2df(map):
    return pd.DataFrame.from_records(map, index=np.ones(1, dtype=int))


def load_mva(filenames, label_feat, mc_feats,
             where='/', tabname='mva', n_events=None, n_events_per_file=None,
             from_beginning=True, shuffle_rows=False):
    """
    A Loader for "stupid" MVA files (all in single table).

    Useful, for example, if you have files output by `rootpy.root2hdf5`.

    Parameters
    ----------
    from_beginning: bool, default=True
        if true, slice file[:n_events]. if false, file[n_events:]
    n_events / n_events_per_file:
        if n_events_per_file is set, ignore n_events
        if both are unset, read all events from each file
    shuffle_rows: bool, default=False
        Whether to shuffle the sample positions around. useful if your data is
        chunked (likely when reading multipple files.
        e.g. [000011112222] -> [022001201112]
    """
    if shuffle_rows:
        from sklearn.utils import shuffle
    df = []
    for fn in filenames:
        if n_events_per_file:
            n = n_events_per_file[fn]
        else:
            n = n_events
        with tb.open_file(fn, 'r') as h5:
            tab = h5.get_node(where, tabname)
            if not n:
                buf = tab[:]
            else:
                if from_beginning:
                    buf = tab[:n]
                else:
                    buf = tab[n:]
            df.append(buf)
    df = np.concatenate(df)
    feats = df.dtype.names
    reco_feats = [feat for feat in feats
                  if feat not in mc_feats and feat != label_feat]
    X_reco = df[reco_feats]
    mc_info = df[mc_feats]
    y = df[label_feat]
    if shuffle_rows:
        X_reco, mc_info, y = shuffle(X_reco, mc_info, y)
    return X_reco, y, mc_info



def open_hdf5(filename):
    return read_hdf5(filename)


def read_hdf5(filename, detx=None, det_id=None, det_from_file=False):
    """Open HDF5 file and retrieve all relevant information.

    Optionally, a detector geometry can read by passing a detector file,
    or retrieved from the database by passing a detector ID, or by reading
    the detector id from the event info in the file.
    """
    event_info = read_table(filename, '/event_info')
    hits = read_table(filename, '/hits')
    mc_tracks = read_table(filename, '/mc_tracks')
    reco = read_group(filename, '/reco')
    geometry = read_geometry(detx, det_id, det_from_file,
                             det_id_table=event_info['det_id'])
    return Run(event_info, geometry, hits, mc_tracks, reco)


def read_group(h5file, where):
    if isinstance(h5file, string_types):
        h5file = tb.open_file(h5file, 'r')
    tabs = []
    for table in h5file.iter_nodes(where, classname='Table'):
        tabname = table.name
        tab = table[:]
        tab = insert_prefix_to_dtype(tab, tabname)
        tab = pd.DataFrame.from_records(tab)
        tabs.append(tab)
    h5file.close()
    tabs = pd.concat(tabs, axis=1)
    return tabs


def read_table(filename, where):
    with tb.open_file(filename, 'r') as h5:
        tab = h5.get_node(where)[:]
    tab = pd.DataFrame.from_records(tab)
    return tab


def write_table(filename, where, array):
    """Write a numpy array into a H5 table."""
    filt = tb.Filters(complevel=5, shuffle=True, fletcher32=True)
    loc, tabname = os.path.split(where)
    with tb.open_file(filename, 'a') as h5:
        h5.create_table(loc, tabname, obj=array, createparents=True,
                        filters=filt)
