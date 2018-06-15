# -*- coding: utf-8 -*-
"""
=============================
Reading and Parsing EVT files
=============================

This example shows how to read and parse EVT files, which are used in our
Monte Carlo productions.
"""
from __future__ import absolute_import, print_function, division

# Author: Tamas Gal <tgal@km3net.de>, Moritz Lotze >mlotze@km3net.de>
# License: BSD-3
import matplotlib.pyplot as plt
import numpy as np

from km3modules.common import StatusBar
from km3pipe import Module, Pipeline
from km3pipe.dataclasses import Table
from km3pipe.calib import Calibration
from km3pipe.io import EvtPump
from km3pipe.math import pld3
import km3pipe.style

km3pipe.style.use("km3pipe")

filename = "../data/numu_cc.evt"
detx = "../data/km3net_jul13_90m_r1494_corrected.detx"


class VertexHitDistanceCalculator(Module):
    """Calculate vertex-hit-distances"""

    def configure(self):
        self.distances = []

    def process(self, blob):
        tracks = blob['TrackIns']
        muons = tracks[tracks.type == 5]
        muon = Table(muons[np.argmax(muons.energy)])
        hits = blob['CalibHits']
        dist = pld3(hits.pos, muon.pos, muon.dir)
        self.distances.append(dist)
        return blob

    def finish(self):
        dist_flat = np.concatenate(self.distances)
        plt.hist(dist_flat)
        plt.savefig('dists.pdf')


pipe = Pipeline()
pipe.attach(EvtPump, filename=filename, parsers=['km3'])
pipe.attach(StatusBar, every=100)
pipe.attach(Calibration, filename=detx)
pipe.attach(VertexHitDistanceCalculator)
pipe.drain(5)
