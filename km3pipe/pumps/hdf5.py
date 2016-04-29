# coding=utf-8
# Filename: hdf5.py
# pylint: disable=C0103,R0903
# vim:set ts=4 sts=4 sw=4 et:
"""
Pumps for the EVT simulation dataformat.

"""
from __future__ import division, absolute_import, print_function

from collections import defaultdict
import os.path

try:
    import numpy as np
except ImportError:
    print("The HDF5 Sink needs numpy: pip install numpy")

try:
    import h5py
except ImportError:
    print("The HDF5 Sink and Bucket need h5py: pip install h5py")

try:
    import tables
except ImportError:
    print("The HDF5 Sink and Bucket need pytables: pip install pytables")

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
        # ignore evt_info so far
        if 'Hits' in blob:
            self._write_hits(blob['Hits'], table_name='hits')

        if 'MCHits' in blob:
            self._write_hits(blob['MCHits'], table_name='mc_hits')

        if 'MCTracks' in blob:
           self._write_tracks(blob['MCTracks'], table_name='mc_tracks')

        self.index += 1
        return blob

    def finish(self):
        self.h5_file.close()


class HDF5TablePump(Pump):
    """Provides a pump for KM3NeT HDF5 files"""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename')
        if os.path.isfile(self.filename):
            self._h5file = tables.File(self.filename)
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

    def _get_hits(self, index, where):
        hits = {}
        hits['channel_id'] = where.channel_id[index]
        hits['dom_id'] = where.dom_id[index]
        hits['id'] = where.id[index]
        hits['pmt_id'] = where.pmt_id[index]
        hits['time'] = where.time[index]
        hits['tot'] = where.tot[index]
        hits['triggered'] = where.triggered[index]
        return HitSeries.from_dict(hits)

    def _get_tracks(self, index, where):
        tracks = {}
        tracks['dir'] = where.dir[index]
        tracks['energy'] = where.energy[index]
        tracks['id'] = where.id[index]
        tracks['pos'] = where.pos[index]
        tracks['time'] = where.time[index]
        tracks['type'] = where.type[index]
        return TrackSeries.from_dict(tracks)

    def get_blob(self, index):
        blob = {}
        n_event = index + 1
        blob['Hits'] = self._get_hits(index, self.h5file.root.hits)
        blob['MCHits'] = self._get_hits(index, self.h5file.root.mc_hits)
        blob['MCTracks'] = self._get_tracks(index, self.h5file.root.mc_tracks)
        return blob

    def finish(self):
        """Clean everything up"""
        self._h5file.close()

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
