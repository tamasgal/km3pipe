#!/usr/bin/env python
"""
======================
Basic Analysis Example
======================

"""
# Author: Tamás Gál <tgal@km3net.de>
# License: BSD-3
# Date: 2017-10-10
# Status: Under construction...

#####################################################
# Preparation
# -----------
# The very first thing we do is importing our libraries and setting up the Jupyter Notebook environment.

import matplotlib.pyplot as plt   # our plotting module
import pandas as pd               # the main HDF5 reader
import numpy as np                # must have
import km3pipe as kp              # some KM3NeT related helper functions
import seaborn as sb              # beautiful statistical plots!


#####################################################
# this is just to make our plots a bit "nicer", you can skip it
import km3pipe.style
km3pipe.style.use("km3pipe")


#####################################################
# Accessing the Data File(s)
# --------------------------
# In the following, we will work with one random simulation file with
# reconstruction information from JGandalf which has been converted
# from ROOT to HDF5 using the ``tohdf5`` command line tool provided by
# ``KM3Pipe``.
#
# You can find the documentation here: http://km3pipe.readthedocs.io/en/latest/cmd.html#tohdf

#####################################################
# Note for Lyon Users
# ~~~~~~~~~~~~~~~~~~~
# If you are working on the Lyon cluster, you can activate the latest KM3Pipe
# with the following command (put it in your ``~/.bashrc`` to load it
# automatically in each shell session)::
#
#     source /afs/in2p3.fr/throng/km3net/src/python/pyenv.sh

#####################################################
# Converting from ROOT to HDF5 (if needed)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Choose a file (take e.g. one from /in2p3/km3net/mc/...),
# load the appropriate Jpp/Aanet version and convert it via::
#
#     tohdf5 --aa-format=the_file.root --ignore-hits --skip-header
#
# Note that you may have to ``--skip-header`` otherwise you might
# encounter some segfaults. There is currently a big mess of different
# versions of libraries in several levels of the MC file processings.
#
# The ``--ignore-hits`` will skip the hit information, so the converted file
# is much smaller (normally around 2-3 MBytes). Skip this option if you want
# to read the hit information too. The file will still be smaller than the
# ROOT file (about 1/3).
#
# Luckily, a handful people are preparing the HDF5 conversion, so in future
# you can download them directly, without thinking about which Jpp or Aanet
# version you need to open them.

#####################################################
# First Look at the Data
# ----------------------

filepath = "data/basic_analysis_sample.h5"


#####################################################
# We can have a quick look at the file with the ``ptdump`` command in the terminal::
#
#     ptdump filename.h5
#
# For further information, check out the documentation of the KM3NeT HDF5
# format definition: http://km3pipe.readthedocs.io/en/latest/hdf5.html
#


#####################################################
# The ``/event_info`` table contains general information about each event.
# The data is a simple 2D table and each event is represented by a single row.
#
# Let's have a look at the first few rows:
event_info = pd.read_hdf(filepath, '/event_info')
print(event_info.head(5))


#####################################################
# Next, we will read out the MC tracks which are stored under ``/mc_tracks``.

tracks = pd.read_hdf(filepath, '/mc_tracks')

#####################################################
# also read event info, for things like weights

info = pd.read_hdf(filepath, '/event_info')

#####################################################
# It has a similar structure, but now you can have multiple rows which belong
# to an event. The ``event_id`` column holds the ID of the corresponding event.

print(tracks.head(10))

#####################################################
# We now are accessing the first track for each event by grouping via
# ``event_id`` and calling the ``first()`` method of the
# ``Pandas.DataFrame`` object.

primaries = tracks.groupby('event_id').first()

#####################################################
# Here are the first 5 primaries:
print(primaries.head(5))

#####################################################
# Creating some Fancy Graphs
# --------------------------

#####################################################
#
primaries.energy.hist(bins=100, log=True)
plt.xlabel('energy [GeV]')
plt.ylabel('number of events')
plt.title('Energy Distribution');


#####################################################
#
primaries.bjorkeny.hist(bins=100)
plt.xlabel('bjorken-y')
plt.ylabel('number of events')
plt.title('bjorken-y Distribution');


#####################################################
#
zeniths = kp.math.zenith(primaries.filter(regex='^dir_.?$'))
primaries['zenith'] = zeniths

plt.hist(np.cos(primaries.zenith), bins=21, histtype='step', linewidth=2)
plt.xlabel(r'cos($\theta$)')
plt.ylabel('number of events')
plt.title('Zenith Distribution');


#####################################################
#
# Starting positions of primaries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.hist2d(primaries.pos_x, primaries.pos_y, bins=100);
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('2D Plane')
plt.colorbar();

#####################################################
#
# If you have seaborn installed (`pip install seaborn`), you can easily create
# nice jointplots:
try:
    import seaborn as sns
    km3pipe.style.use("km3pipe")  # reset matplotlib style
except:
    print("No seaborn found, skipping example.")
else:
    g = sns.jointplot('pos_x', 'pos_y', data=primaries, kind='hex')
    g.set_axis_labels("x [m]", "y[m]")


#####################################################
#
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(primaries.pos_x, primaries.pos_y, primaries.pos_z, s=3)
ax.set_xlabel('x [m]', labelpad=10)
ax.set_ylabel('y [m]', labelpad=10)
ax.set_zlabel('z [m]', labelpad=10)
ax.set_title('3D Plane');


#####################################################
#
gandalfs = pd.read_hdf(filepath, '/reco/gandalf')
print(gandalfs.head(5))


#####################################################
#
gandalfs.columns


#####################################################
#
plt.hist(gandalfs['lambda'], bins=50, log=True);
plt.xlabel('lambda parameter')
plt.ylabel('count')
plt.title('Lambda Distribution of Reconstructed Events');


#####################################################
#
gandalfs['zenith'] = kp.math.zenith(gandalfs.filter(regex='^dir_.?$'))

plt.hist((gandalfs.zenith - primaries.zenith).dropna(), bins=100)
plt.xlabel(r'true zenith - reconstructed zenith [rad]')
plt.ylabel('count')
plt.title('Zenith Reconstruction Difference');


#####################################################
#
l = 5
lambda_cut = gandalfs['lambda'] < l
plt.hist((gandalfs.zenith - primaries.zenith)[lambda_cut].dropna(), bins=100)
plt.xlabel(r'true zenith - reconstructed zenith [rad]')
plt.ylabel('count')
plt.title('Zenith Reconstruction Difference for lambda < {}'.format(l));


#####################################################
# let's fit some distributions
# gaussian + lorentz (aka norm + cauchy)

from scipy.stats import cauchy, norm

residuals = gandalfs.zenith - primaries.zenith
cut = (gandalfs['lambda'] < l) & (np.abs(residuals) < 2 * np.pi)
residuals = residuals[cut]
info[cut]

# convert rad -> deg
residuals = residuals * 180 / np.pi

pi = 180
# x axis for plotting
x = np.linspace(-pi, pi, 1000)

c_loc, c_gamma = cauchy.fit(residuals)
g_mu, g_sigma = norm.fit(residuals)

plt.hist(residuals, bins='auto', label='Histogram', normed=True)
plt.plot(x, cauchy(c_loc, c_gamma).pdf(x),
         label='Lorentz: $\gamma =${:.3f}'.format(c_gamma))
plt.plot(x, norm(g_mu, g_sigma).pdf(x),
         label='Gauss: $\sigma =$ {:.3f}'.format(g_sigma))
plt.xlim(-pi/4, pi/4)
plt.xlabel('Zenith residuals / deg')
plt.legend()

#####################################################
# let's get the Median Absolute Deviation

resid_median = np.median(residuals)
residuals_shifted_by_median = residuals - resid_median
absolute_deviation = np.abs(residuals_shifted_by_median)
resid_mad = np.median(absolute_deviation)

plt.hist(np.abs(residuals), alpha=.7, bins='auto')
plt.axvline(resid_mad)
plt.title("Average resolution: {:.3f} degree".format(resid_mad))
