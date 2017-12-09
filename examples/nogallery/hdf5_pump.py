#!/usr/bin/env python
from km3pipe import Module, Pipeline
from km3pipe.io import HDF5Pump


class Printer(Module):
    def process(self, blob):
        print(blob['HitTable']['dom_id'])
        return blob


FILENAME = '/Users/tamasgal/Data/KM3NeT/DU-2/KM3NeT_00000007_00001597.root.h5'

pipe = Pipeline()
pipe.attach(HDF5Pump, filename=FILENAME)
pipe.attach(Printer)
pipe.drain(5)
