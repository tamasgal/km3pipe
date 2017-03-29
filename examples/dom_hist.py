"""
==================
DOM hits.
==================

Estimate track/DOM distances using the number of hits per DOM.
"""

# Author: Tamas Gal <tgal@km3net.de>
# License: BSD-3

from collections import defaultdict, Counter

import km3pipe as kp
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from km3modules.common import StatusBar
from km3pipe.math import pld3
import km3pipe.style
km3pipe.style.use("km3pipe")


filename = "data/km3net_jul13_90m_muatm50T655.km3_v5r1.JTE_r2356.root.0-499.h5"
geo = kp.Geometry(filename="data/km3net_jul13_90m_r1494_corrected.detx")


def filter_muons(blob):
    """Write all muons from McTracks to Muons."""
    tracks = blob['McTracks']
    muons = [t for t in tracks if t.type == 5]
    blob["Muons"] = kp.dataclasses.McTrackSeries(muons)
    return blob


class DOMHits(kp.Module):
    """Create histogram with n_hits and distance of hit to track."""
    def configure(self):
        self.hit_statistics = defaultdict(list)

    def process(self, blob):
        hits = blob['Hits']
        #muons = blob['Muons']
        muons = blob['McTracks']

        highest_energetic_muon = max(muons, key=lambda x: x.energy)
        muon = highest_energetic_muon

        triggered_hits = hits.triggered_hits
        dom_hits = Counter(triggered_hits.dom_id)
        for dom_id, n_hits in dom_hits.items():
            distance = pld3(geo.detector.dom_positions[dom_id],
                            muon.pos,
                            muon.dir)
            self.hit_statistics['n_hits'].append(n_hits)
            self.hit_statistics['distance'].append(distance)
        return blob

    def finish(self):
        df = pd.DataFrame(self.hit_statistics)
        sdf = df[(df['distance'] < 200) & (df['n_hits'] < 50)]
        plt.hist2d(sdf['distance'], sdf['n_hits'], cmap='plasma',
                   bins=(max(sdf['distance']) - 1, max(sdf['n_hits']) - 1),
                   norm=LogNorm())
        plt.xlabel('Distance between hit and muon track [m]')
        plt.ylabel('Number of hits on DOM')
        plt.show()


pipe = kp.Pipeline()
pipe.attach(kp.io.HDF5Pump, filename=filename)
pipe.attach(StatusBar, every=100)
pipe.attach(filter_muons)
pipe.attach(DOMHits)
pipe.drain()
