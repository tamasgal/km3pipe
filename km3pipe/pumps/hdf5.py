# coding=utf-8
# Filename: hdf5.py
# pylint: disable=C0103,R0903
"""
Pumps for the EVT simulation dataformat.

"""
from __future__ import division, absolute_import, print_function

import os.path

try:
    import pandas as pd
except ImportError:
    print("The HDF5 pump needs pandas: pip install pandas")

try:
    import h5py
except ImportError:
    print("The HDF5 sink needs h5py: pip install h5py")


from km3pipe import Pump, Module
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = 'tamasgal'


class HDF5Pump(Pump):
    """Provides a pump for KM3NeT HDF5 files"""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename')
        self.create_hit_list = self.get('create_hit_list') or True
        if os.path.isfile(self.filename):
            self._store = pd.HDFStore(self.filename)
            self._hits_by_event = self._store.hits.groupby('event_id')
            self._n_events = len(self._hits_by_event)
        else:
            raise IOError("No such file or directory: '{0}'"
                          .format(self.filename))
        self.index = 0

    def process(self, blob):
        try:
            blob = self.get_blob(self.index)
        except KeyError:
            self.index = 0
            raise StopIteration
        self.index += 1
        return blob

    def get_blob(self, index):
        hits = self._hits_by_event.get_group(index)
        blob = {'HitTable': hits}
        if self.create_hit_list:
            blob['Hits'] = list(hits.itertuples())
        return blob

    def finish(self):
        """Clean everything up"""
        self._store.close()

    def __len__(self):
        return self._n_events

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        try:
            blob = self.get_blob(self.index)
        except KeyError:
            self.index = 0
            raise StopIteration
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
        self.mc_tracks = {}
        self.index = 0
        print("Processing {0}...".format(self.filename))

    def process(self, blob):
        try:
            self._add_hits(blob['Hits'])
        except KeyError:
            print("No hits found. Skipping...")

        try:
            self._add_mc_tracks(blob['MCTracks'])
        except KeyError:
            print("No MC tracks found. Skipping...")

        self.index += 1
        return blob

    def _add_hits(self, hits):
        for hit in hits:
            self.hits.setdefault('event_id', []).append(self.index)
            self.hits.setdefault('id', []).append(hit.id)
            self.hits.setdefault('pmt_id', []).append(hit.pmt_id)
            self.hits.setdefault('time', []).append(hit.t)
            self.hits.setdefault('tot', []).append(hit.tot)
            self.hits.setdefault('triggered', []).append(bool(hit.trig))
            self.hits.setdefault('dom_id', []).append(hit.dom_id)
            self.hits.setdefault('channel_id', []).append(ord(hit.channel_id))

    def _add_mc_tracks(self, mc_tracks):
        for mc_track in mc_tracks:
            self.mc_tracks.setdefault('event_id', []).append(self.index)
            self.mc_tracks.setdefault('id', []).append(mc_track.id)
            self.mc_tracks.setdefault('x', []).append(mc_track.pos.x)
            self.mc_tracks.setdefault('y', []).append(mc_track.pos.y)
            self.mc_tracks.setdefault('z', []).append(mc_track.pos.z)
            self.mc_tracks.setdefault('dx', []).append(mc_track.dir.x)
            self.mc_tracks.setdefault('dy', []).append(mc_track.dir.y)
            self.mc_tracks.setdefault('dz', []).append(mc_track.dir.z)
            self.mc_tracks.setdefault('time', []).append(mc_track.t)
            self.mc_tracks.setdefault('energy', []).append(mc_track.E)
            self.mc_tracks.setdefault('type', []).append(mc_track.type)

    def finish(self):
        h5_file = h5py.File(self.filename, 'w')
        if self.hits:
            df = pd.DataFrame(self.hits)
            rec = df.to_records(index=False)
            h5_file.create_dataset('/hits', data=rec)
            print("Finished writing hits in {0}".format(self.filename))
        if self.mc_tracks:
            df = pd.DataFrame(self.mc_tracks)
            rec = df.to_records(index=False)
            h5_file.create_dataset('/mc_tracks', data=rec)
            print("Finished writing MC tracks in {0}".format(self.filename))
        h5_file.close()
