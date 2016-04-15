#!/usr/bin/env python
from km3pipe import Pipeline, Geometry, Module
from km3pipe.pumps import EvtPump
import os
import time

PATH = '/Users/tamasgal/Data/KM3NeT'
DETX = 'Detector/km3net_jul13_90m_r1494_corrected.detx'
DATA = 'km3net_jul13_90m_muatm10T23.km3_v5r1.JTE.evt'


class PrintPositions(Module):
    def process(self, blob):
        print("Hit positions:")
        print(blob['Hits'].pos)
        return blob

    def finish(self):
        time.sleep(2)


pipe = Pipeline(timeit=True)
pipe.attach(EvtPump, filename=os.path.join(PATH, DATA))
pipe.attach(Geometry, apply=True, filename=os.path.join(PATH, DETX))
pipe.attach(PrintPositions, timeit=True)
pipe.drain(3)
