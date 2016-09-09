# coding=utf-8
# Filename: hdf5.py
# pylint: disable=C0103,R0903
# vim:set ts=4 sts=4 sw=4 et:
"""
Read and write KM3NeT-formatted HDF5 files.
"""
from __future__ import division, absolute_import, print_function

from collections import defaultdict
import os.path
from six import string_types

import pandas as pd
import tables as tb

import km3pipe as kp
from km3pipe import Pump, Module
from km3pipe.dataclasses import ArrayTaco, deserialise_map
from km3pipe.logger import logging
from km3pipe.tools import camelise, decamelise, insert_prefix_to_dtype, split

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"

FORMAT_VERSION = '1.0'
MINIMUM_FORMAT_VERSION = '1.0'


class HDF5Sink(Module):
    """Write KM3NeT-formatted HDF5 files, event-by-event.

    The data can be a numpy structured array, a pandas DataFrame,
    or a km3pipe dataclass object with a `serialise()` method.

    The name of the corresponding H5 table is the decamelised
    blob-key, so values which are stored in the blob under `FooBar`
    will be written to `/foo_bar` in the HDF5 file.

    To store at a different location in the file, the data needs a
    `.h5loc` attribute:

    >>> my_arr.h5loc = '/somewhere'

    Parameters
    ----------
    filename: str, optional (default: 'dump.h5')
        Where to store the events.
    """
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename') or 'dump.h5'
        self.index = 1
        self.h5file = tb.open_file(self.filename, mode="w", title="KM3NeT")
        self.filters = tb.Filters(complevel=5, shuffle=True,
                                  fletcher32=True)
        self._tables = {}

    def _to_array(self, data):
        if len(data) <= 0:
            return
        try:
            return data.to_records()
        except AttributeError:
            pass
        try:
            return data.serialise()
        except AttributeError:
            pass
        return data

    def _write_array(self, where, arr, title=''):
        if where not in self._tables:
            dtype = arr.dtype
            loc, tabname = os.path.split(where)
            self._tables[where] = self.h5file.create_table(
                loc, tabname, description=dtype, title=title,
                filters=self.filters, createparents=True)
        tab = self._tables[where]
        tab.append(arr)

    def process(self, blob):
        for key, entry in sorted(blob.items()):
            if hasattr(entry, 'dtype') or hasattr(entry, 'serialise') or \
                    hasattr(entry, 'to_records'):
                try:
                    h5loc = entry.h5loc
                except AttributeError:
                    h5loc = '/'
                where = os.path.join(h5loc, decamelise(key))
                entry = self._to_array(entry)
                if entry is None:
                    continue
                self._write_array(where, entry, title=key)

        if not self.index % 1000:
            for tab in self._tables.values():
                tab.flush()

        self.index += 1
        return blob

    def finish(self):
        for tab in self._tables.values():
            tab.cols.event_id.create_index()
            tab.flush()
        self.h5file.root._v_attrs.km3pipe = str(kp.__version__)
        self.h5file.root._v_attrs.pytables = str(tb.__version__)
        self.h5file.root._v_attrs.format_version = str(FORMAT_VERSION)
        self.h5file.close()


class HDF5Pump(Pump):
    """Read KM3NeT-formatted HDF5 files, event-by-event.

        Parameters
        ----------
        filename: str
        From where to read events.
        """
    def __init__(self, filename, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = filename
        if os.path.isfile(self.filename):
            self.h5_file = tb.File(self.filename)
            if not self.get("no_version_check"):
                self._check_version()
        else:
            raise IOError("No such file or directory: '{0}'"
                          .format(self.filename))
        self.index = None
        self._reset_index()

        try:
            event_info = self.h5_file.get_node('/', 'event_info')
            self.event_ids = event_info.cols.event_id[:]
        except tb.NoSuchNodeError:
            log.critical("No /event_info table found.")
            raise SystemExit

        self._n_events = len(self.event_ids)

    def _check_version(self):
        try:
            version = str(self.h5_file.root._v_attrs.format_version)
        except AttributeError:
            log.error("Could not determine HDF5 format version, you may "
                      "encounter unexpected errors! Good luck...")
            return

        if split(version, int, '.') < split(MINIMUM_FORMAT_VERSION, int, '.'):
            raise SystemExit("HDF5 format version {0} or newer required!\n"
                             "'{1}' has HDF5 format version {2}."
                             .format(MINIMUM_FORMAT_VERSION,
                                     self.filename,
                                     version))

    def process(self, blob):
        try:
            blob = self.get_blob(self.index)
        except KeyError:
            self._reset_index()
            raise StopIteration
        self.index += 1
        return blob

    def get_blob(self, index):
        event_id = self.event_ids[index]
        blob = {}
        for tab in self.h5_file.walk_nodes(classname="Table"):
            loc, tabname = os.path.split(tab._v_pathname)
            tabname = camelise(tabname)
            try:
                dc = deserialise_map[tabname]
            except KeyError:
                dc = ArrayTaco
            arr = tab.read_where('event_id == %d' % event_id)
            blob[tabname] = dc.deserialise(arr, h5loc=loc, event_id=event_id)
        return blob

    def finish(self):
        """Clean everything up"""
        self.h5_file.close()

    def _reset_index(self):
        """Reset index to default value"""
        self.index = 0

    def __len__(self):
        return self._n_events

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        if self.index >= self._n_events:
            self._reset_index()
            raise StopIteration
        blob = self.get_blob(self.index)
        self.index += 1
        return blob

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_blob(index)
        elif isinstance(index, slice):
            return self._slice_generator(index)
        else:
            raise TypeError("index must be int or slice")

    def _slice_generator(self, index):
        """A simple slice generator for iterations"""
        start, stop, step = index.indices(len(self))
        for i in range(start, stop, step):
            yield self.get_blob(i)


class H5Chain(object):
    """Read/write multiple HDF5 files as ``pandas.DataFrame``.

    Parameters
    ----------
    which: list(str) or dict(str->cond)
        The filenames to be read in. When passing a dict, events are
        selected according to cond, which can be `None` (all events), a
        slice, or a numexpr-like pytables condition string.

    Examples
    --------
    >>> filenames = ['numu_cc.h5', 'anue_nc.h5']
    >>> c = H5Chain(filenames)

    specify n_events per file, or their event ids

    >>> filenames = {'numu_cc.h5': None, 'anue_nc.h5': 100,
                 'numu_cc.h5': [1, 2, 3],}
    >>> c = H5Chain(filenames)

    these are pandas Dataframes

    >>> X = c.reco
    >>> wgt = c.event_info.weights_w2
    >>> Y_ene = c.mc_tracks[::2].energy

    """

    def __init__(self, filenames, table_filter=None):
        if table_filter is None:
            table_filter = {}
        if isinstance(filenames, list):
            filenames = {key: None for key in filenames}
        self._which = filenames
        self._store = defaultdict(list)

        for fil, cond in sorted(self._which.items()):
            h5fil = tb.open_file(fil, 'r')

            # tables under '/', e.g. mc_tracks
            for tab in h5fil.iter_nodes('/', classname='Table'):
                if tab.name in table_filter.keys():
                    tab_cond = table_filter[tab.name]
                    arr = self._read_table(tab, tab_cond)
                else:
                    arr = self._read_table(tab, cond)
                arr = pd.DataFrame.from_records(arr)
                self._store[tab.name].append(arr)

            # groups under '/', e.g. '/reco'
            for gr in h5fil.iter_nodes('/', classname='Group'):
                arr = self._read_group(gr, cond)
                self._store[gr._v_name].append(arr)

            h5fil.close()

        for key, dfs in sorted(self._store.items()):
            self._store[key] = pd.concat(dfs)

        for key, val in sorted(self._store.items()):
            setattr(self, key, val)

    def __getitem__(self, name):
        return self._store[name]

    def _read_group(cls, group, cond):
        # Store through groupname, insert tablename into dtype
        tabs = []
        for tab in group._f_iter_nodes(classname='Table'):
            tabname = tab.name
            arr = cls._read_table(tab, cond)
            arr = insert_prefix_to_dtype(arr, tabname)
            arr = pd.DataFrame.from_records(arr)
            tabs.append(arr)
        tabs = pd.concat(tabs, axis=1)
        return tabs

    @classmethod
    def _read_table(cls, table, cond=None):
        if cond is None:
            return table[:]
        if isinstance(cond, string_types):
            return table.read_where(cond)
        if isinstance(cond, int):
            return table[:cond]
        if isinstance(cond, slice):
            return table[cond]
        return table.read(cond)
