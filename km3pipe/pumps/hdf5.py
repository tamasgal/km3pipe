# coding=utf-8
# Filename: hdf5.py
# pylint: disable=C0103,R0903
# vim:set ts=4 sts=4 sw=4 et:
"""
Pumps for the EVT simulation dataformat.

"""
from __future__ import division, absolute_import, print_function

import os.path

import tables
import numpy as np

from km3pipe import Pump, Module
from km3pipe.dataclasses import HitSeries, TrackSeries
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = 'tamasgal'

POS_ATOM = tables.FloatAtom(shape=3)


class Hit(tables.IsDescription):
    channel_id = tables.UInt8Col()
    dom_id = tables.UIntCol()
    event_id = tables.UIntCol()
    id = tables.UIntCol()
    pmt_id = tables.UIntCol()
    time = tables.IntCol()
    tot = tables.UInt8Col()
    triggered = tables.BoolCol()


class Track(tables.IsDescription):
    dir = tables.FloatCol(shape=(3,))
    energy = tables.FloatCol()
    event_id = tables.UIntCol()
    id = tables.UIntCol()
    pos = tables.FloatCol(shape=(3,))
    time = tables.IntCol()
    type = tables.IntCol()


class HDF5Sink2(Module):
    def __init__(self, **context):
        """A Module to convert (KM3NeT) ROOT files to HDF5."""
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename') or 'dump.h5'
        self.index = 1
        self.h5file = tables.open_file(self.filename, mode="w", title="Test file")
        self.hits = self.h5file.create_table('/', 'hits', Hit, "Hits")
        self.mc_hits = self.h5file.create_table('/', 'mc_hits', Hit, "MC Hits")
        self.mc_tracks = self.h5file.create_table('/', 'mc_tracks', Track, "MC Tracks")

    def _write_hits(self, hits, hit_row):
        for hit in hits:
            hit_row['channel_id'] = hit.channel_id
            hit_row['dom_id'] = hit.dom_id
            hit_row['event_id'] = self.index
            hit_row['id'] = hit.id
            hit_row['pmt_id'] = hit.pmt_id
            hit_row['time'] = hit.time
            hit_row['tot'] = hit.tot
            hit_row['triggered'] = hit.triggered
            hit_row.append()

    def _write_tracks(self, tracks, track_row):
        for track in tracks:
            track_row['dir'] = track.dir
            track_row['energy'] = track.energy
            track_row['event_id'] = self.index
            track_row['id'] = track.id
            track_row['pos'] = track.pos
            track_row['time'] = track.time
            track_row['type'] = track.type
            track_row.append()

    def process(self, blob):
        hits = blob['Hits']
        self._write_hits(hits, self.hits.row)
        try:
            mc_hits = blob['MCHits']
            mc_tracks = blob['MCTracks']
        except KeyError:
            pass
        else:
            self._write_hits(mc_hits, self.mc_hits.row)
            self._write_tracks(mc_tracks, self.mc_tracks.row)

        if not self.index % 1000:
            self.hits.flush()
            self.mc_hits.flush()
            self.mc_tracks.flush()

        self.index += 1
        return blob

    def finish(self):
        self.hits.flush()
        self.mc_hits.flush()
        self.mc_tracks.flush()
        self.h5file.close()


class HDF5Sink(Module):
    def __init__(self, **context):
        """A Module to convert (KM3NeT) ROOT files to HDF5."""
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename') or 'dump.h5'
        self.complevel = self.get('complevel') or 5
        self.filter = tables.Filters(complevel=self.complevel)
        self.h5_file = tables.File(self.filename, 'w')
        self.index = 1
        self._prepare_hits()
        self._prepare_hits(group_name='mc_hits')
        self._prepare_tracks(group_name='mc_tracks')
        self._prepare_event_info(group_name='info')
        print("Processing {0}...".format(self.filename))

    def _prepare_event_info(self, group_name='info', where='/'):
        info_group = self.h5_file.create_group(where, group_name)
        h5_file = self.h5_file
        h5_file.create_earray(info_group, 'id', atom=tables.IntAtom(), shape=(0, ), filters=self.filter)
        h5_file.create_earray(info_group, 'det_id', atom=tables.IntAtom(), shape=(0, ), filters=self.filter)
        h5_file.create_earray(info_group, 'frame_index', atom=tables.UIntAtom(), shape=(0, ), filters=self.filter)
        h5_file.create_earray(info_group, 'mc_id', atom=tables.IntAtom(), shape=(0, ), filters=self.filter)
        h5_file.create_earray(info_group, 'mc_t', atom=tables.Float64Atom(), shape=(0, ), filters=self.filter)
        h5_file.create_earray(info_group, 'overlays', atom=tables.UInt8Atom(), shape=(0, ), filters=self.filter)
        h5_file.create_earray(info_group, 'run_id', atom=tables.UIntAtom(), shape=(0, ), filters=self.filter)
        #h5_file.create_earray(info_group, 'timestamp', atom=tables.Float64Atom(), shape=(0, ), filters=self.filter)
        h5_file.create_earray(info_group, 'trigger_counter', atom=tables.UInt64Atom(), shape=(0, ), filters=self.filter)
        h5_file.create_earray(info_group, 'trigger_mask', atom=tables.UInt64Atom(), shape=(0, ), filters=self.filter)

    def _prepare_hits(self, group_name='hits', where='/'):
        hit_group = self.h5_file.create_group(where, group_name)
        h5_file = self.h5_file
        h5_file.create_vlarray(hit_group, 'channel_id', atom=tables.UInt8Atom(), filters=self.filter)
        h5_file.create_vlarray(hit_group, 'dom_id', atom=tables.UIntAtom(), filters=self.filter)
        h5_file.create_vlarray(hit_group, 'id', atom=tables.UIntAtom(), filters=self.filter)
        h5_file.create_vlarray(hit_group, 'pmt_id', atom=tables.UIntAtom(), filters=self.filter)
        h5_file.create_vlarray(hit_group, 'time', atom=tables.IntAtom(), filters=self.filter)
        h5_file.create_vlarray(hit_group, 'tot', atom=tables.UInt8Atom(), filters=self.filter)
        h5_file.create_vlarray(hit_group, 'triggered', atom=tables.BoolAtom(), filters=self.filter)

    def _prepare_tracks(self, group_name='tracks', where='/'):
        track_group = self.h5_file.create_group(where, group_name)
        h5_file = self.h5_file
        h5_file.create_vlarray(track_group, 'dir', atom=POS_ATOM, filters=self.filter)
        h5_file.create_vlarray(track_group, 'energy', atom=tables.FloatAtom(), filters=self.filter)
        h5_file.create_vlarray(track_group, 'id', atom=tables.UIntAtom(), filters=self.filter)
        h5_file.create_vlarray(track_group, 'pos', atom=POS_ATOM, filters=self.filter)
        h5_file.create_vlarray(track_group, 'time', atom=tables.IntAtom(), filters=self.filter)
        h5_file.create_vlarray(track_group, 'type', atom=tables.IntAtom(), filters=self.filter)

    def _write_event_info(self, evt, table_name='info', where='/'):
        target = self.h5_file.get_node(where, table_name)
        target.id.append(np.array([evt.id, ]))
        target.det_id.append(np.array([evt.det_id, ]))
        target.run_id.append(np.array([evt.run_id, ]))
        target.frame_index.append(np.array([evt.frame_index, ]))
        target.trigger_mask.append(np.array([evt.trigger_mask, ]))
        target.trigger_counter.append(np.array([evt.trigger_counter, ]))
        target.overlays.append(np.array([evt.overlays, ]))
        #target.timestamp.append(np.array([evt.timestamp, ]))
        target.mc_id.append(np.array([evt.mc_id, ]))
        target.mc_t.append(np.array([evt.mc_t, ]))

    def _write_hits(self, hits, table_name='hits', where='/'):
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
        if 'Hits' in blob:
            self._write_hits(blob['Hits'], table_name='hits')

        if 'MCHits' in blob:
            self._write_hits(blob['MCHits'], table_name='mc_hits')

        if 'MCTracks' in blob:
            self._write_tracks(blob['MCTracks'], table_name='mc_tracks')

        if 'Evt' in blob:
            self._write_event_info(blob['Evt'], table_name='info')

        self.index += 1
        return blob

    def finish(self):
        self.h5_file.close()


class HDF5Pump(Pump):
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
        #info['timestamp'] = table.timestamp[index]
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
