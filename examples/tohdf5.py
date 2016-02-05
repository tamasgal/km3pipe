#!/usr/bin/env python
"""
Converts hits in a Jpp-ROOT file to HDF5.

"""
from km3pipe.pumps.aanet import AanetPump
from km3pipe import Pipeline, Module

import sys
import pandas as pd

if len(sys.argv) < 3:
    sys.exit('Usage: {0} FILENAME.root OUTPUTFILENAME.h5'.format(sys.argv[0]))

FILEPATH = sys.argv[1]
OUTPUTFILEPATH = sys.argv[2]


class HDF5Sink(Module):
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename') or 'dump.h5'
        self.hits = {}
        self.index = 0
        print("Processing {0}...".format(self.filename))

    def process(self, blob):
        for hit in blob['Hits']:
            self.hits.setdefault('event_id', []).append(self.index)
            self.hits.setdefault('id', []).append(hit.id)
            self.hits.setdefault('pmt_id', []).append(hit.pmt_id)
            self.hits.setdefault('time', []).append(hit.t)
            self.hits.setdefault('tot', []).append(ord(hit.tot))
            self.hits.setdefault('triggered', []).append(bool(hit.trig))
            self.hits.setdefault('dom_id', []).append(hit.dom_id)
            self.hits.setdefault('channel_id', []).append(ord(hit.channel_id))
        self.index += 1
        return blob

    def finish(self):
        if self.hits:
            df = pd.DataFrame(self.hits)
            df.to_hdf(self.filename, 'hits', format='table')
            print("Finished {0}".format(self.filename))
        else:
            print("Skipping {0}".format(self.filename))


pipe = Pipeline()
pipe.attach(AanetPump, filename=FILEPATH)
pipe.attach(HDF5Sink, filename=OUTPUTFILEPATH)
pipe.drain()
