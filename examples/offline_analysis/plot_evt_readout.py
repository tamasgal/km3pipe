# -*- coding: utf-8 -*-
"""
=============================
Reading and Parsing EVT files
=============================

This example shows how to read and parse EVT files, which are used in our
Monte Carlo productions.
"""

# Author: Tamas Gal <tgal@km3net.de>, Moritz Lotze >mlotze@km3net.de>
# License: BSD-3
import matplotlib.pyplot as plt
import numpy as np

import km3pipe as kp
import km3modules as km
from km3net_testdata import data_path

kp.style.use("km3pipe")

filename = data_path("evt/example_numuCC.evt")
detx = data_path("detx/km3net_jul13_90m_r1494_corrected.detx")


class VertexHitDistanceCalculator(kp.Module):
    """Calculate vertex-hit-distances"""

    def configure(self):
        self.distances = []

    def process(self, blob):
        tracks = blob["TrackIns"]
        muons = tracks[tracks.type == 5]
        muon = kp.Table(muons[np.argmax(muons.energy)])
        hits = blob["CalibHits"]
        dist = kp.math.pld3(hits.pos, muon.pos, muon.dir)
        self.distances.append(dist)
        return blob

    def finish(self):
        dist_flat = np.concatenate(self.distances)
        plt.hist(dist_flat)
        plt.xlabel("distance between hits and muon / m")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig("dists.png")


pipe = kp.Pipeline()
pipe.attach(kp.io.EvtPump, filename=filename, parsers=["km3"])
pipe.attach(km.StatusBar, every=100)
pipe.attach(kp.calib.Calibration, filename=detx)
pipe.attach(VertexHitDistanceCalculator)
pipe.drain(5)
