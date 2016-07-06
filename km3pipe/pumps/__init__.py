# coding=utf-8
# Filename: __init__.py
"""
A collection of pumps for different kinds of data formats.

"""
from __future__ import division, absolute_import, print_function

import os

from km3pipe.pumps.evt import EvtPump  # noqa
from km3pipe.pumps.daq import DAQPump  # noqa
from km3pipe.pumps.clb import CLBPump  # noqa
from km3pipe.pumps.aanet import AanetPump  # noqa
from km3pipe.pumps.jpp import JPPPump  # noqa
from km3pipe.pumps.ch import CHPump  # noqa
from km3pipe.pumps.hdf5 import HDF5Pump  # noqa
from km3pipe.pumps.hdf5 import HDF5Sink  # noqa
from km3pipe.pumps.pickle import PicklePump  # noqa

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

    pumps = {
            '.evt': EvtPump,
            '.h5': HDF5Pump,
            '.aa.root': AanetPump,
            '.root': JPPPump if use_jppy else AanetPump,
            '.dat': DAQPump,
            '.dqd': CLBPump,
            }

    if extension not in pumps:
        log.critical("No pump found for '{0}'".format(extension))

    return pumps[extension](filename=filename, name=name)


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
