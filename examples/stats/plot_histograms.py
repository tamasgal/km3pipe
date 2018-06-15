# -*- coding: utf-8 -*-
"""
==========
Histograms
==========

Load a histogram from a file, plot it, draw random samples.
"""
from __future__ import absolute_import, print_function, division

# Author: Moritz Lotze <mlotze@km3net.de>
# License: BSD-3

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns

import km3pipe.style.moritz    # noqa

#####################################################
# Load the histogram from a file.
# a histogram is just bincounts + binlimits.

filename = "../data/hist_example.h5"

with h5py.File(filename, 'r') as f:
    counts = f['/hist/counts'][:]
    binlims = f['/hist/binlims'][:]

print(counts)
print(counts.shape)
print(binlims)
print(binlims.shape)

#####################################
# create a distribution object

hist = scipy.stats.rv_histogram((counts, binlims))

#####################################
# plot it

# make an x axis for plotting
padding = 3
n_points = 10000
x = np.linspace(binlims[0] - padding, binlims[-1] + padding, n_points)

plt.plot(x, hist.pdf(x))

#####################################
# plot the cumulative histogram

plt.plot(x, hist.cdf(x))

##########################################################
# sample from the histogram (aka draw random variates)

n_sample = 30
sample = hist.rvs(size=n_sample)

#############################################################################
# let's plot it (use seaborn to plot the data points as small vertical bars)
plt.hist(sample, bins='auto', alpha=.5)
sns.rugplot(sample, color='k', linewidth=3)
