"""
==================
zt-plots.
==================

This example shows how to create a zt-plot of a given DU and event ID.
"""

# Author: Tamas Gal <tgal@km3net.de>
# License: BSD-3

import pandas as pd
import matplotlib.pyplot as plt
import km3pipe as kp
import km3pipe.style
km3pipe.style.use("km3pipe")


DU = 26
EVENT_ID = 23
filename = "data/km3net_jul13_90m_muatm50T655.km3_v5r1.JTE_r2356.root.0-499.h5"
geometry = kp.Geometry(filename="data/km3net_jul13_90m_r1494_corrected.detx")

all_hits = pd.read_hdf(filename, 'hits', mode='r')
hits = all_hits[all_hits.event_id == EVENT_ID].copy()
geometry.apply(hits)

fig, ax = plt.subplots()

hits[hits['du'] == DU].plot('time', 'pos_z', style='.', ax=ax, label='hits')
triggered_hits = hits[(hits['du'] == DU) & (hits['triggered'] == True)]
triggered_hits.plot('time', 'pos_z', style='.', ax=ax, label='triggered hits')

ax.set_title("zt-plot of event {0} on DU{1}".format(EVENT_ID, DU))
ax.set_xlabel("time [ns]")
ax.set_ylabel("z [m]")
