"""
==================
zt-plots.
==================

This example shows how to create a zt-plot of a given DU and event ID.
"""

# Author: Tamas Gal <tgal@km3net.de>
# License: BSD-3

import matplotlib.pyplot as plt
import km3pipe as kp
import km3pipe.style
km3pipe.style.use("km3pipe")

DU = 26
EVENT_ID = 23
filename = "data/km3net_jul13_90m_muatm50T655.km3_v5r1.JTE_r2356.root.0-499.h5"
cal = kp.calib.Calibration(
    filename="data/km3net_jul13_90m_r1494_corrected.detx"
)
pump = kp.io.hdf5.HDF5Pump(filename=filename)

raw_hits = pump[EVENT_ID]["Hits"]
hits = cal.apply(raw_hits).conv_to("pandas")

fig, ax = plt.subplots()

ax.scatter(hits.time, hits.pos_z, label='hits')

ax.set_title("zt-plot of event {0} on DU{1}".format(EVENT_ID, DU))
ax.set_xlabel("time [ns]")
ax.set_ylabel("z [m]")
