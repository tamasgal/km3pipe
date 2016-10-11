# coding=utf-8
# Filename: __init__.py
"""
A collection of io for different kinds of data formats.

"""
from __future__ import division, absolute_import, print_function

import os.path
from six import string_types

import numpy as np
import pandas as pd
import tables as tb

from km3pipe import Geometry, Run
from km3pipe.io.evt import EvtPump  # noqa
from km3pipe.io.daq import DAQPump  # noqa
from km3pipe.io.clb import CLBPump  # noqa
from km3pipe.io.aanet import AanetPump  # noqa
from km3pipe.io.jpp import JPPPump  # noqa
from km3pipe.io.ch import CHPump  # noqa
from km3pipe.io.hdf5 import HDF5Pump  # noqa
from km3pipe.io.hdf5 import HDF5Sink  # noqa
from km3pipe.io.hdf5 import H5Chain  # noqa
from km3pipe.io.pickle import PicklePump  # noqa
from km3pipe.tools import insert_prefix_to_dtype

from km3pipe.logger import logging

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal, Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


log = logging.getLogger(__name__)


def GenericPump(filenames, use_jppy=False, name="GenericPump", **kwargs):
    """A generic pump which utilises the appropriate pump."""
    if not isinstance(filenames, string_types):
        fn = filenames[0]
    else:
        fn = filenames
    extension = os.path.splitext(fn)[1]

    io = {
        '.evt': EvtPump,
        '.h5': HDF5Pump,
        '.root': JPPPump if use_jppy else AanetPump,
        '.dat': DAQPump,
        '.dqd': CLBPump,
    }

    if extension not in io:
        log.critical("No pump found for '{0}'".format(extension))

    if isinstance(filenames, string_types):
        return io[extension](filename=filenames, name=name, **kwargs)
    else:
        if len(filenames) == 1:
            return io[extension](filename=filenames[0], name=name, **kwargs)
        return io[extension](filenames=filenames, name=name, **kwargs)


def df_to_h5(df, filename, tabname, filemode='a', where='/', complevel=5,):
    """Write pandas dataframes with proper columns.

    The main 2 ways pandas writes dataframes suck bigly.
    """
    with tb.open_file(filename, filemode) as h5:
        filt = tb.Filters(complevel=complevel, shuffle=True)
        h5.create_table(where, tabname, obj=df.to_records(), filters=filt)


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


def guess_mc_feats(df):
    feats = df.columns
    mc_feats = []
    mc_feats.extend([f for f in feats if 'MC' in f])
    mc_feats.extend([f for f in feats if 'ID' in f])
    mc_feats.extend([f for f in feats if 'weight' in f.lower()])
    mc_feats.extend([f for f in feats if 'nevents' in f.lower()])
    mc_feats.extend([f for f in feats if 'livetime' in f.lower()])
    return mc_feats


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
    reco = _read_group(filename, '/reco')
    geometry = read_geometry(detx, det_id, det_from_file,
                             det_id_table=event_info['det_id'])
    return Run(event_info, geometry, hits, mc_tracks, reco)


def read_geometry(detx=None, det_id=None, from_file=False, det_id_table=None):
    """Retrive geometry from file, the DB."""
    if not detx or det_id or from_file:
        return None
    if detx is not None:
        return Geometry(filename=detx)
    if from_file:
        det_ids = np.unique(det_id_table)
        if len(det_ids) > 1:
            log.critical("Multiple detector IDs found in events.")
        det_id = det_ids[0]
    if det_id is not None:
        if det_id < 0:
            log.warning("Negative detector ID found ({0}), skipping..."
                        .format(det_id))
            return None
        try:
            return Geometry(det_id=det_id)
        except ValueError:
            log.warning("Could not retrieve the geometry information.")
    return None


def _read_group(h5file, where):
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
