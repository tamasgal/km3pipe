#!/usr/bin/env python
from km3pipe import Pipeline, Geometry, Module
from km3pipe.pumps import EvtPump

detx = 'km3net_jul13_90m_r1494_corrected.detx'


class PrintPositions(Module):
    def process(self, blob):
        print("Hit positions:")
        print(blob['Hits'].pos)
        return blob


pipe = Pipeline()
pipe.attach(EvtPump, filename='km3net_jul13_90m_muatm10T23.km3_v5r1.JTE.evt')
pipe.attach(Geometry, apply=True, filename=detx)
pipe.attach(PrintPositions)
pipe.drain(3)
