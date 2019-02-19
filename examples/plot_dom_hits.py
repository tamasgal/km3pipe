# -*- coding: utf-8 -*-
"""
==================
DOM hits.
==================

Estimate track/DOM distances using the number of hits per DOM.
"""
from __future__ import absolute_import, print_function, division

# Author: Tamas Gal <tgal@km3net.de>
# License: BSD-3

from collections import defaultdict, Counter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import km3pipe as kp
from km3pipe.dataclasses import Table
from km3pipe.math import pld3
from km3modules.common import StatusBar
import km3pipe.style
km3pipe.style.use("km3pipe")

filename = "data/atmospheric_muons_sample.h5"
cal = kp.calib.Calibration(filename="data/KM3NeT_-00000001_20171212.detx")


def filter_muons(blob):
    """Write all muons from McTracks to Muons."""
    tracks = blob['McTracks']
    muons = tracks[tracks.type == -13]    # PDG particle code
    blob["Muons"] = Table(muons)
    return blob


class DOMHits(kp.Module):
    """Create histogram with n_hits and distance of hit to track."""

    def configure(self):
        self.hit_statistics = defaultdict(list)

    def process(self, blob):
        hits = blob['Hits']
        muons = blob['Muons']

        highest_energetic_muon = Table(muons[np.argmax(muons.energy)])
        muon = highest_energetic_muon

        triggered_hits = hits.triggered_rows

        dom_hits = Counter(triggered_hits.dom_id)
        for dom_id, n_hits in dom_hits.items():
            try:
                distance = pld3(
                    cal.detector.dom_positions[dom_id], muon.pos, muon.dir
                )
            except KeyError:
                self.log.warning("DOM ID %s not found!" % dom_id)
                continue
            self.hit_statistics['n_hits'].append(n_hits)
            self.hit_statistics['distance'].append(distance)
        return blob

    def finish(self):
        df = pd.DataFrame(self.hit_statistics)
        print(df)
        sdf = df[(df['distance'] < 200) & (df['n_hits'] < 50)]
        bins = (max(sdf['distance']) - 1, max(sdf['n_hits']) - 1)
        plt.hist2d(
            sdf['distance'],
            sdf['n_hits'],
            cmap='plasma',
            bins=bins,
            norm=LogNorm()
        )
        plt.xlabel('Distance between hit and muon track [m]')
        plt.ylabel('Number of hits on DOM')
        plt.show()


pipe = kp.Pipeline()
pipe.attach(kp.io.HDF5Pump, filename=filename)
pipe.attach(StatusBar, every=100)
pipe.attach(filter_muons)
pipe.attach(DOMHits)
pipe.drain()
