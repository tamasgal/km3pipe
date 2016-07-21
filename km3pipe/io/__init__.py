# coding=utf-8
# Filename: __init__.py
"""
A collection of io for different kinds of data formats.

"""
from __future__ import division, absolute_import, print_function

import os

from km3pipe.io.evt import EvtPump  # noqa
from km3pipe.io.daq import DAQPump  # noqa
from km3pipe.io.clb import CLBPump  # noqa
from km3pipe.io.aanet import AanetPump  # noqa
from km3pipe.io.jpp import JPPPump  # noqa
from km3pipe.io.ch import CHPump  # noqa
from km3pipe.io.hdf5 import HDF5Pump  # noqa
from km3pipe.io.hdf5 import HDF5Sink  # noqa
from km3pipe.io.pickle import PicklePump  # noqa

from km3pipe.logger import logging

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal, Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


log = logging.getLogger(__name__)


def GenericPump(filename, use_jppy=False, name="GenericPump"):
    """A generic pump which utilises the appropriate pump."""
    extension = os.path.splitext(filename)[1]

    io = {
            '.evt': EvtPump,
            '.h5': HDF5Pump,
            '.aa.root': AanetPump,
            '.root': JPPPump if use_jppy else AanetPump,
            '.dat': DAQPump,
            '.dqd': CLBPump,
            }

    if extension not in io:
        log.critical("No pump found for '{0}'".format(extension))

    return io[extension](filename=filename, name=name)


def df_to_h5(df, filename, tabname, filemode='a', where='/', complevel=5,):
    """Write pandas dataframes with proper columns.

    The main 2 ways pandas writes dataframes suck bigly.
    """
    from tables import Filters, open_file
    with open_file(filename, filemode) as h5:
        filt = Filters(complevel=complevel, shuffle=True)
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
    import pandas as pd
    from tables import open_file
    if shuffle_rows:
        from sklearn.utils import shuffle
    df = []
    for fn in filenames:
        if n_events_per_file:
            n = n_events_per_file[fn]
        else:
            n = n_events
        with open_file(fn, 'r') as h5:
            tab = h5.get_node(where, tabname)
            if not n:
                buf = tab[:]
            else:
                if from_beginning:
                    buf = tab[:n]
                else:
                    buf = tab[n:]
            buf = pd.DataFrame.from_records(buf)
            df.append(buf)
    df = pd.concat(df)
    feats = df.columns
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


def read_hdf5(filename, detx=None, det_id=None, ignore_geometry=False):
    """Open HDF5 file and retrieve all relevant information."""
    event_info = pd.read_hdf(filename, '/event_info')
    geometry = None
    hits = pd.read_hdf(filename, '/hits')
    mc_tracks = pd.read_hdf(filename, '/mc_tracks')
    try:
        reco = read_reco(filename)
    except ValueError:
        reco = None

    if not ignore_geometry:
        if detx is not None:
            geometry = kp.Geometry(filename=detx)
        if det_id is not None:
            geometry = kp.Geoemtry(det_id=det_id)

        if detx is None and det_id is None:
            det_ids = np.unique(event_info.det_id)
            if len(det_ids) > 1:
                log.critical("Multiple detector IDs found in events.")
            det_id = det_ids[0]
            if det_id > 0:
                try:
                    geometry = kp.Geometry(det_id=det_id)
                except ValueError:
                    log.warning("Could not retrieve the geometry information.")
            else:
                log.warning("Negative detector ID found ({0}), skipping..."
                            .format(det_id))

    return kp.Run(event_info, geometry, hits, mc_tracks, reco)


def read_reco(filename):
    df = []
    with pd.HDFStore(filename, 'r') as h5:
        reco_group = h5.get_node('/reco')
        for table in reco_group:
            tabname = table.name
            buf = table[:]
            new_names = [tabname + '_' + col for col in buf.dtype.names]
            buf.dtype.names = new_names
            buf = pd.DataFrame.from_records(buf)
            df.append(buf)
    df = pd.concat(df, axis=1)
    return df

