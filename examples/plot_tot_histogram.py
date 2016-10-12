"""
==================
ToT histogram.
==================

This example shows how to create a simple histogram of the PMT signals.
"""

import pandas as pd
import matplotlib.pyplot as plt
import km3pipe.style


filename = "data/km3net_jul13_90m_muatm50T655.km3_v5r1.JTE_r2356.root.0-499.h5"

hits = pd.read_hdf(filename, 'hits')
hits.hist("tot", bins=254, log=True, edgecolor='none')
plt.title("ToT distribution")
plt.xlabel("ToT [ns]")
