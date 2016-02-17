#!/usr/bin/env python
"""
Converts hits in a Jpp-ROOT file to HDF5.

"""
from km3pipe.pumps.aanet import AanetPump
from km3pipe import Pipeline, Module

import sys
import pandas as pd
import h5py

if len(sys.argv) < 3:
    sys.exit('Usage: {0} FILENAME.root OUTPUTFILENAME.h5'.format(sys.argv[0]))

FILEPATH = sys.argv[1]
OUTPUTFILEPATH = sys.argv[2]


class HDF5Sink(Module):
    def __init__(self, **context):
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
            self.hits.setdefault('tot', []).append(ord(hit.tot))
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
        h5 = h5py.File(self.filename, 'w')
        if self.hits:
            df = pd.DataFrame(self.hits)
            rec = df.to_records(index=False)
            h5.create_dataset('/hits', data=rec)
            print("Finished writing hits in {0}".format(self.filename))
        if self.mc_tracks:
            df = pd.DataFrame(self.mc_tracks)
            rec = df.to_records(index=False)
            h5.create_dataset('/mc_tracks', data=rec)
            print("Finished writing MC tracks in {0}".format(self.filename))
        h5.close()


pipe = Pipeline()
pipe.attach(AanetPump, filename=FILEPATH)
pipe.attach(HDF5Sink, filename=OUTPUTFILEPATH)
pipe.drain()
