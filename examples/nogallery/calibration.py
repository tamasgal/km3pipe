#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calibration

"""
from __future__ import absolute_import, print_function, division

from km3pipe import Pipeline, Module
from km3pipe.calib import Calibration
from km3pipe.io import EvtPump
from km3modules.common import StatusBar
import os
import time

PATH = '/Users/tamasgal/Data/KM3NeT'
DETX = 'Detector/km3net_jul13_90m_r1494_corrected.detx'
DATA = 'km3net_jul13_90m_muatm10T23.km3_v5r1.JTE.evt'


class PrintPositions(Module):
    def process(self, blob):
        print("Hit positions:")
        print(blob['Hits'].pos)
        time.sleep(0.1)
        return blob

    def finish(self):
        time.sleep(2)


pipe = Pipeline()
pipe.attach(EvtPump, filename=os.path.join(PATH, DATA))
pipe.attach(StatusBar)
pipe.attach(
    Calibration, timeit=True, apply=True, filename=os.path.join(PATH, DETX)
)
pipe.attach(PrintPositions, timeit=True)
pipe.drain(3)
