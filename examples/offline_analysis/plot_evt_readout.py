"""
=============================
Reading and Parsing EVT files
=============================

Work in progress!

This example shows how to read and parse EVT files, which are used in our
Monte Carlo productions.

"""
# Author: Tamas Gal <tgal@km3net.de>
# License: BSD-3
import numpy as np
import km3pipe as kp
from km3pipe.dataclasses import Table
from km3modules.common import StatusBar
import km3pipe.style
km3pipe.style.use("km3pipe")


filename = "../data/numu_cc.evt"
detx = "../data/km3net_jul13_90m_r1494_corrected.detx"

cal = kp.calib.Calibration(filename=detx)


class VertexHitDistanceCalculator(kp.Module):
    """Calculate vertex-hit-distances"""

    def process(self, blob):
        print(blob.keys())
        print(blob['raw_header']['seed'])
        # tracks = blob['McTracks']
        # muons = tracks[tracks.type == 5]
        # muon = Table(muons[np.argmax(muons.energy)])
        # print(muon)
        return blob


pipe = kp.Pipeline()
pipe.attach(kp.io.EvtPump, filename=filename, parsers='auto')
pipe.attach(StatusBar, every=100)
pipe.attach(VertexHitDistanceCalculator)
pipe.drain(2)
