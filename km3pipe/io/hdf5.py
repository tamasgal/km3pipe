# coding=utf-8
# Filename: hdf5.py
# pylint: disable=C0103,R0903
# vim:set ts=4 sts=4 sw=4 et:
"""
Pumps for the EVT simulation dataformat.

"""
from __future__ import division, absolute_import, print_function

import os.path

import numpy as np
import tables

import km3pipe
from km3pipe import Pump, Module
from km3pipe.dataclasses import HitSeries, TrackSeries, EventInfo, Reco
from km3pipe.logger import logging
from km3pipe.tools import camelise, decamelise

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class HDF5Sink(Module):
    def __init__(self, **context):
        """A Module to dump blob data to HDF5.

        Each item in the blob which has a dtype or a `serialise()` method
        will be dumped in the HDF5 file.
        The table name is either the key (if the value is a plain value with
        a dtype) or the `loc` attribute of the object.

        The target table name is a decamelised version of the blob-key,
        so for example values which are stored in the blob under `FooBar`
        will be written to `/foo_bar` in the HDF5 file.
        """
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename') or 'dump.h5'
        self.index = 1
        self.h5file = tables.open_file(self.filename, mode="w", title="KM3NeT")
        self.filters = tables.Filters(complevel=5, shuffle=True,
                                      fletcher32=True)
        self._tables = {}

    def _write_table(self, where, data, title=''):
        if where not in self._tables:
            dtype = data.dtype
            loc, tabname = os.path.split(where)
            self._tables[where] = self.h5file.create_table(
                loc, tabname, description=dtype, title=title,
                filters=self.filters, createparents=True)

        tab = self._tables[where]
        try:
            data = data.serialise()
        except AttributeError:
            pass
        if len(data) <= 0:
            return
        tab.append(data)

    def process(self, blob):
        for key, entry in blob.items():
            if hasattr(entry, 'dtype') or hasattr(entry, 'serialise'):
                try:
                    loc = entry.loc
                except AttributeError:
                    loc = ''
                where = loc + '/' + decamelise(key)
                self._write_table(where, entry, title=key)

        if not self.index % 1000:
            for tab in self._tables.values():
                tab.flush()

        self.index += 1
        return blob

    def finish(self):
        for tab in self._tables.values():
            tab.cols.event_id.create_index()
        self.h5file.root._v_attrs.km3pipe = str(km3pipe.__version__)
        self.h5file.root._v_attrs.pytables = str(tables.__version__)
        self.h5file.close()


class HDF5Pump(Pump):
    """Provides a pump for KM3NeT HDF5 files"""
    def __init__(self, filename, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = filename
        if os.path.isfile(self.filename):
            self.h5_file = tables.File(self.filename)
        else:
            raise IOError("No such file or directory: '{0}'"
                          .format(self.filename))
        self.index = None
        self._reset_index()

        try:
            event_info = self.h5_file.get_node('/', 'event_info')
            self.event_ids = event_info.cols.event_id[:]
        except tables.NoSuchNodeError:
            log.critical("No /event_info table found.")
            raise SystemExit

        self._n_events = len(self.event_ids)

    def process(self, blob):
        try:
            blob = self.get_blob(self.index)
        except KeyError:
            self._reset_index()
            raise StopIteration
        self.index += 1
        return blob

    def _get_event(self, event_id, where):
        if not where.startswith('/'):
            where = '/' + where
        table = self.h5_file.get_node(where)
        return table.read_where('event_id == %d' % event_id)

    def _get_group(self, event_id, group_name='reco', where='/'):
        group = self.h5_file.get_node(where, group_name)
        out = {}
        for table in group:
            tabname = table.title
            data = table.read_where('event_id == %d' % event_id)
            out[tabname] = data
        return out

    def get_blob(self, index):
        event_id = self.event_ids[index]
        blob = {}
        blob['Hits'] = HitSeries.from_table(
            self._get_event(event_id, where='hits'))
        blob['MCHits'] = HitSeries.from_table(
            self._get_event(event_id, where='mc_hits'))
        blob['MCTracks'] = TrackSeries.from_table(
            self._get_event(event_id, where='mc_tracks'))
        blob['EventInfo'] = EventInfo.from_table(
            self._get_event(event_id, where='event_info'))
        blob['EventInfo'] = self._get_event_info(event_id,
                                                 where='event_info')
        reco = RecoSeries()
        reco_path = '/reco'
        reco_group = self.h5_file.get_node(reco_path)
        for recname in reco_group._v_children.keys():
            loc = '/'.join(reco_path, recname)
            reco[recname] = self._get_event(event_id, where=loc)
        blob['Reco'] = reco

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
