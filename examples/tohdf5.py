#!/usr/bin/env python
from km3pipe.pumps.aanet import AanetPump
from km3pipe import Pipeline, Module

import pandas as pd

FILEPATH='KM3NeT_00000007_00000160.root'


class HDF5Sink(Module):
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename') or 'dump.h5'
        self.hits = {}

    def process(self, blob):
        print("Processing blob...")
        for hit in blob['Hits']:
            self.hits.setdefault('id', []).append(hit.id)
            self.hits.setdefault('pmt_id', []).append(hit.pmt_id)
            self.hits.setdefault('time', []).append(hit.t)
            self.hits.setdefault('tot', []).append(ord(hit.tot))
            self.hits.setdefault('trig', []).append(bool(hit.trig))
            self.hits.setdefault('dom_id', []).append(hit.dom_id)
            self.hits.setdefault('channel_id', []).append(ord(hit.channel_id))
        return blob

    def finish(self):
        df = pd.DataFrame(self.hits)
        df.to_hdf(self.filename, 'hits')


pipe = Pipeline()
pipe.attach(AanetPump, filename=FILEPATH)
pipe.attach(HDF5Sink, filename=FILEPATH + '.h5')
pipe.drain()
