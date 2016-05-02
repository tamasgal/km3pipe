# coding=utf-8
# Filename: hdf5.py
# pylint: disable=C0103,R0903
# vim:set ts=4 sts=4 sw=4 et:
"""
Pumps for the EVT simulation dataformat.

"""
from __future__ import division, absolute_import, print_function

import os.path
from collections import OrderedDict

import tables
import numpy as np

from km3pipe import Pump, Module
from km3pipe.dataclasses import HitSeries, TrackSeries
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = 'tamasgal'

POS_ATOM = tables.FloatAtom(shape=3)


class HDF5TableSink(Module):
    def __init__(self, **context):
        """A Module to convert (KM3NeT) ROOT files to HDF5."""
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename') or 'dump.h5'
        self.h5_file = tables.File(self.filename, 'w')
        self.index = 1
        self._prepare_hits()
        self._prepare_hits(group_name='mc_hits')
        self._prepare_tracks(group_name='mc_tracks')
        self._prepare_event_info(group_name='mc_tracks')
        print("Processing {0}...".format(self.filename))

    def _prepare_hits(self, group_name='hits', where='/'):
        hit_group = self.h5_file.create_group(where, group_name)
        h5_file = self.h5_file
        h5_file.create_vlarray(hit_group, 'channel_id', atom=tables.UInt8Atom())
        h5_file.create_vlarray(hit_group, 'dom_id', atom=tables.UIntAtom())
        h5_file.create_vlarray(hit_group, 'id', atom=tables.UIntAtom())
        h5_file.create_vlarray(hit_group, 'pmt_id', atom=tables.UIntAtom())
        h5_file.create_vlarray(hit_group, 'time', atom=tables.IntAtom())
        h5_file.create_vlarray(hit_group, 'tot', atom=tables.UInt8Atom())
        h5_file.create_vlarray(hit_group, 'triggered', atom=tables.BoolAtom())

    def _prepare_tracks(self, group_name='tracks', where='/'):
        track_group = self.h5_file.create_group(where, group_name)
        h5_file = self.h5_file
        h5_file.create_vlarray(track_group, 'dir', atom=POS_ATOM)
        h5_file.create_vlarray(track_group, 'energy', atom=tables.FloatAtom())
        h5_file.create_vlarray(track_group, 'id', atom=tables.UIntAtom())
        h5_file.create_vlarray(track_group, 'pos', atom=POS_ATOM)
        h5_file.create_vlarray(track_group, 'time', atom=tables.IntAtom())
        h5_file.create_vlarray(track_group, 'type', atom=tables.IntAtom())

    def _prepare_event_info(self):
        # Postpone array creation to finish(),
        # since we don't know n_events from the start
        self.id = []
        self.det_id = []
        self.mc_id = []
        self.run_id = []
        self.trigger_mask = []
        self.trigger_counter = []
        self.overlays = []
        self.timestamp = []
        self.mc_t = []
        # Ignore these AANet-user constructs so far
        # str comment
        # int index
        # int flags

    def _get_event_info(self, evt):
        self.id.append(evt.id)
        self.det_id.append(evt.det_id)
        self.mc_id.append(evt.mc_id)
        self.run_id.append(evt.run_id)
        self.trigger_mask.append(evt.trigger_mask)
        self.trigger_counter.append(evt.trigger_counter)
        self.overlays.append(evt.overlays)
        self.timestamp.append(evt.timestamp)
        # is this always available, I wonder?
        self.mc_t.append(evt.mc_t)

    def _write_event_info(self, group_name='info', where='/'):
        mg = self.h5_file.create_group(where, group_name)
        # order is the same as in AAnet evt.h
        h5.create_array(mg, 'det_id', np.array(self.det_id, dtype=int))
        h5.create_array(mg, 'mc_id', np.array(self.mc_id, dtype=int))
        h5.create_array(mg, 'run_id', np.array(self.run_id, dtype=int))
        h5.create_array(mg, 'frame_index', np.array(self.frame_index, dtype=int))
        h5.create_array(mg, 'trigger_mask', np.array(self.trigger_mask, dtype=np.dtype('u8')))
        h5.create_array(mg, 'trigger_counter', np.array(self.trigger_counter, dtype=np.dtype('u8')))
        h5.create_array(mg, 'overlays', np.array(self.overlays, dtype=np.dtype('u4')))
        h5.create_array(mg, 'timestamp', np.array(self.timestamp, dtype=float))
        h5.create_array(mg, 'mc_t', np.array(self.mc_t, dtype=float))
        # Ignore these AANet-user constructs so far
        # str comment
        # int index
        # int flags

    def _write_hits(self, hits, table_name='hits', where='/'):
        print(table_name)
        print(hits)
        target = self.h5_file.get_node(where, table_name)
        target.channel_id.append(hits.channel_id)
        target.dom_id.append(hits.dom_id)
        target.id.append(hits.id)
        target.pmt_id.append(hits.pmt_id)
        target.time.append(hits.time)
        target.tot.append(hits.tot)
        target.triggered.append(hits.triggered)

    def _write_tracks(self, tracks, table_name='tracks', where='/'):
        target = self.h5_file.get_node(where, table_name)
        target.dir.append(tracks.dir)
        target.energy.append(tracks.energy)
        target.id.append(tracks.id)
        target.pos.append(tracks.pos)
        target.time.append(tracks.time)
        target.type.append(tracks.type)

    def process(self, blob):
        # ignore evt_info so far
        if 'Hits' in blob:
            self._write_hits(blob['Hits'], table_name='hits')

        if 'MCHits' in blob:
            self._write_hits(blob['MCHits'], table_name='mc_hits')

        if 'MCTracks' in blob:
            self._write_tracks(blob['MCTracks'], table_name='mc_tracks')

        if 'EventInfo' in blob:
            self._get_event_info(blob['EventInfo'], table_name='info')

        self.index += 1
        return blob

    def finish(self):
        self._get_write_info()
        self.h5_file.close()


class HDF5TablePump(Pump):
    """Provides a pump for KM3NeT HDF5 files"""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename')
        if os.path.isfile(self.filename):
            self.h5_file = tables.File(self.filename)
        else:
            raise IOError("No such file or directory: '{0}'"
                          .format(self.filename))
        self.index = None
        self._reset_index()

    def process(self, blob):
        try:
            blob = self.get_blob(self.index)
        except KeyError:
            self._reset_index()
            raise StopIteration
        self.index += 1
        return blob

    def _get_hits(self, index, table_name='hits', where='/'):
        table = self.h5_file.get_node(where, table_name)
        _channel_id = table.channel_id[index]
        _dom_id = table.dom_id[index]
        _id = table.id[index]
        _pmt_id = table.pmt_id[index]
        _time = table.time[index]
        _tot = table.tot[index]
        _triggered = table.triggered[index]
        return HitSeries.from_arrays(_channel_id, _dom_id, _id, _pmt_id,
                                     _time, _tot, _triggered)

    def _get_tracks(self, index, table_name='tracks', where='/'):
        table = self.h5_file.get_node(where, table_name)
        _dir = table.dir[index]
        _energy = table.energy[index]
        _id = table.id[index]
        _pos = table.pos[index]
        _time = table.time[index]
        _type = table.type[index]
        return TrackSeries.from_arrays(_dir, _energy, _id, _pos,
                                      _time, _type)

    def _get_event_info(self, index, table_name='info', where='/'):
        table = self.h5_file.get_node(where, table_name)
        info = {}
        info['id'] = table.id[index]
        info['det_id'] = table.det_id[index]
        info['mc_id'] = table.mc_id[index]
        info['run_id'] = table.run_id[index]
        info['trigger_mask'] = table.trigger_mask[index]
        info['trigger_counter'] = table.trigger_counter[index]
        info['overlays'] = table.overlays[index]
        info['timestamp'] = table.timestamp[index]
        info['mc_t'] = table.mc_t[index]
        return info

    def get_blob(self, index):
        blob = {}
        blob['Hits'] = self._get_hits(index, table_name='hits')
        blob['MCHits'] = self._get_hits(index, table_name='mc_hits')
        blob['MCTracks'] = self._get_tracks(index, table_name='mc_tracks')
        blob['EventInfo'] = self._get_event_info(index, table_name='info')
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
