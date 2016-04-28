# coding=utf-8
# Filename: hdf5.py
# pylint: disable=C0103,R0903
"""
Pumps for the EVT simulation dataformat.

"""
from __future__ import division, absolute_import, print_function

from collections import defaultdict
import os.path

try:
    import numpy as np
except ImportError:
    print("The HDF5 Bucket needs numpy: pip install numpy")

try:
    import h5py
    import tables
except ImportError:
    print("The HDF5 Sink and Bucket need h5py: pip install h5py")


from km3pipe import Pump, Module
from km3pipe.dataclasses import HitSeries
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = 'tamasgal'


class HDF5Pump(Pump):
    """Provides a pump for KM3NeT HDF5 files"""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename')
        if os.path.isfile(self.filename):
            self._h5file = h5py.File(self.filename)
            try:
                evt_group = self._h5file['/event']
            except KeyError:
                raise KeyError("No events found.")
            try:
                self._n_events = evt_group.attrs['n_events']
            except KeyError:
                self._n_events = len(evt_group)
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

    def get_blob(self, index):
        blob = {}
        n_event = index + 1
        raw_hits = self._h5file.get('/event/{0}/hits'.format(n_event))
        blob['Hits'] = HitSeries.from_hdf5(raw_hits)
        blob['EventInfo'] = self._h5file.get('/event/{0}/info'.format(n_event))
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


class HDF5Sink(Module):
    def __init__(self, **context):
        """A Module to convert (KM3NeT) ROOT files to HDF5."""
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename') or 'dump.h5'
        self.hits = {}
        self.mc_hits = {}
        self.mc_tracks = {}
        self.event_info = {}
        self.h5_file = h5py.File(self.filename, 'w')
        self.index = 1
        print("Processing {0}...".format(self.filename))

    def process(self, blob):
        target = '/event/{0}/'.format(self.index)
        self.h5_file.create_group(target)

        self._add_event_info(blob, target=target+'info')

        if 'Hits' in blob:
            self._dump_hits(blob['Hits'], target=target+'hits')

        if 'MCHits' in blob:
            self._add_hits(blob['MCHits'], target=target+'mc_hits')

        if 'MCTracks' in blob:
            self._add_tracks(blob['MCTracks'], target=target+'mc_tracks')

        self.index += 1
        return blob

    def _add_event_info(self, blob, target):
        evt = blob['Evt']

        timestamp = evt.t.AsDouble()
        det_id = evt.det_id
        mc_id = evt.mc_id
        mc_t = evt.mc_t
        run = evt.run_id
        overlays = evt.overlays
        trigger_counter = evt.trigger_counter
        trigger_mask = evt.trigger_mask
        frame_index = evt.frame_index

        info = defaultdict(list)

        info['event_id'].append(self.index)
        info['timestamp'].append(timestamp)
        info['det_id'].append(det_id)
        info['mc_id'].append(mc_id)
        info['mc_t'].append(mc_t)
        info['run'].append(run)
        info['overlays'].append(overlays)
        info['trigger_counter'].append(trigger_counter)
        info['trigger_mask'].append(trigger_mask)
        info['frame_index'].append(frame_index)

        self._dump_dict(info, target)

    def _dump_hits(self, hits, target):
        for hit in hits:
            self.h5_file.create_dataset('time', data=hit.time)
            self.h5_file.create_dataset('triggered', data=hit.triggered)
            self.h5_file.create_dataset('tot', data=hit.tot)
            self.h5_file.create_dataset('dom_id', data=hit.dom_id)
            self.h5_file.create_dataset('pmt_id', data=hit.pmt_id)
            self.h5_file.create_dataset('channel_id', data=hit.channel_id)

    def _add_tracks(self, tracks, target):
        tracks_dict = defaultdict(list)
        for track in tracks:
            tracks_dict['id'].append(track.id)
            tracks_dict['x'].append(track.pos.x)
            tracks_dict['y'].append(track.pos.y)
            tracks_dict['z'].append(track.pos.z)
            tracks_dict['dx'].append(track.dir.x)
            tracks_dict['dy'].append(track.dir.y)
            tracks_dict['dz'].append(track.dir.z)
            tracks_dict['time'].append(track.t)
            tracks_dict['energy'].append(track.E)
            tracks_dict['type'].append(track.type)
        self._dump_dict(tracks_dict, target)

    def _dump_dict(self, data, target):
        if not data:
            return
        for key, vec in data.items():
            arr = np.array(vec)
            self.h5_file.create_dataset(target, data=arr)

    def finish(self):
        self.h5_file.close()


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

    def _prepare_hits(self, group_name='hits'):
        hit_group = self.h5_file.create_group('/', group_name)
        self.h5_file.create_vlarray(hit_group, 'channel_id', atom=tables.IntAtom)
        self.h5_file.create_vlarray(hit_group, 'dir', atom=POS_ATOM)
        self.h5_file.create_vlarray(hit_group, 'dom_id', atom=tables.IntAtom)
        self.h5_file.create_vlarray(hit_group, 'pmt_id', atom=tables.IntAtom)
        self.h5_file.create_vlarray(hit_group, 'pos', atom=POS_ATOM)
        self.h5_file.create_vlarray(hit_group, 'time', atom=tables.FloatAtom)
        self.h5_file.create_vlarray(hit_group, 'tot', atom=tables.FloatAtom)
        self.h5_file.create_vlarray(hit_group, 'triggered', atom=tables.BoolAtom)

    def _prepare_tracks(self, group_name='tracks'):
        track_group = self.h5_file.create_group('/', group_name)
        self.h5_file.create_vlarray(track_group, 'dir', atom=POS_ATOM)
        self.h5_file.create_vlarray(track_group, 'energy', atom=tables.FloatAtom)
        self.h5_file.create_vlarray(track_group, 'id', atom=tables.IntAtom)
        self.h5_file.create_vlarray(track_group, 'pos', atom=POS_ATOM)
        self.h5_file.create_vlarray(track_group, 'time', atom=tables.FloatAtom)
        self.h5_file.create_vlarray(track_group, 'type', atom=tables.IntAtom)

    def _dump_hits(self, hits, target):
        target.channel_id.append(hits.channel_id)
        target.dir.append(hits.dir)
        target.dom_id.append(hits.dom_id)
        target.pmt_id.append(hits.pmt_id)
        target.pos.append(hits.pos)
        target.time.append(hits.time)
        target.tot.append(hits.tot)
        target.triggered.append(hits.triggered)

    def _dump_tracks(self, tracks, target):
        target.dir.append(tracks.dir)
        target.energy.append(tracks.energy)
        target.id.append(tracks.id)
        target.pos.append(tracks.pos)
        target.time.append(tracks.time)
        target.type.append(tracks.type)

    def process(self, blob):
        # ignore evt_info so far
        #self._add_event_info(blob, target=target+'info')
        if 'Hits' in blob:
            self._dump_hits(blob['Hits'], target=self.h5_file.root.hits)

        if 'MCHits' in blob:
            self._dump_hits(blob['MCHits'], target=self.h5_file.root.mc_hits)

        if 'MCTracks' in blob:
            self._dump_tracks(blob['MCTracks'], target=self.h5_file.root.mc_tracks)

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

    def get_blob(self, index):
        blob = {}
        n_event = index + 1
        # TODO
        tot = self._h5file.root.hits.tot[index]
        time = self._h5file.root.hits.time[index]
        blob['Hits'] = HitSeries.from_hdf5(raw_hits)
        blob['EventInfo'] = self._h5file.get('/event/{0}/info'.format(n_event))
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

