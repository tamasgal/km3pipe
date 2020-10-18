#!/usr/bin/env python
# coding: utf-8 -*-
"""

==================
Calibrating Hits
==================

Hits stored in ROOT and HDF5 files are usually not calibrated, which
means that they have invalid positions, directions and uncorrected hit times.
This example shows how to assign the PMT position and direction to each hit
and applying a time correction to them.

The KM3NeT offline format (derived from aanet) uses a single class for every
hit type (regular hits, MC hits and their correspeonding calibrated and
uncalibrated counter parts). In ROOT files, the actual struct/class definition
is stored along the data, which means that all the attributes are accessible
even when they are invalid or not instantiated. Positions and directions are
also part of these attributes and they are initialised to `(0, 0, 0)`.

The `km3pipe.calib.Calibration()` class can be used to load a calibration and
the `.apply()` method to update the position, direction and correct the arrival
time of hit.

"""

# Author: Tamas Gal <tgal@km3net.de>
# License: BSD-3

import km3pipe as kp
import km3io
from km3net_testdata import data_path

#######################
# The `offline/km3net_offline.root` contains 10 events with hit information:

f = km3io.OfflineReader(data_path("offline/km3net_offline.root"))

#######################
# The corresponding calibration file is stored in `detx/km3net_offline.detx`:

calib = kp.calib.Calibration(filename=data_path("detx/km3net_offline.detx"))

#######################
# Let's grab the hits of the event with index `5`:

hits = f.events[5].hits

#######################
# The positions and directions show the default values (0) and are not
# calibrated. Here are the values of the first few hits:
n = 7
for attr in [f"{k}_{q}" for k in ("pos", "dir") for q in "xzy"]:
    print(attr, hits[attr][:n])

#######################
# Here are the uncalibrated times:
uncalibrated_times = hits.t
print(uncalibrated_times[:n])


#######################
# To calibrate the hits, use the `calib.apply()` method which will create a
# `km3pipe.Table`, retrieve the positions and directions of the corresponding
# PMTs, apply the time calibration and also do the PMT time slew correction.

calibrated_hits = calib.apply(hits)

#######################
# The calibrated hits are stored in a `kp.Table` which is a thin wrapper
# around a `numpy.record` array (a simple numpy array with named attributes):

print(calibrated_hits.dtype)


#######################
# The positions and directions are now showing the correct values and the time
# is the calibrated one:

for attr in [f"{k}_{q}" for k in ("pos", "dir") for q in "xzy"]:
    print(attr, calibrated_hits[attr][:n])

print(calibrated_hits.time[:n])

#######################
# The `t0` field holds the time calibration correction which was automatically
# added to hit time (`hit.time`):

print(calibrated_hits.t0[:n])


#######################
# As mentioned above, the PMT time slewing correction is also applied, which
# is a tiny correction of the arrival time with respect to the hit's ToT value.
# We can reveal their values by subtracting the t0 from the calibrated time and
# compare to the uncalibrated ones. Notice that hits represented as `kp.Table`
# in km3pipe use `.time` instead of `.t` for mainly historical reasons:

slews = uncalibrated_times - (calibrated_hits.time - calibrated_hits.t0)
print(slews[:n])

#######################
# Let's compare the slews with the ones calculated with `kp.calib.slew()`. The
# values match very well, with tiny variations due to floating point arithmetic:

slew_diff = slews - kp.calib.slew(hits.tot)
print(slew_diff[:n])

######################
# To omit the PMT slew calibration, you can pass the `correct_slewing=False`
# option the `.apply()`:

calibrated_hits_no_slew_correction = calib.apply(hits, correct_slewing=False)

######################
# The difference between the calibration with and without slewing is obviously
# the slewing correction itself:

print((calibrated_hits_no_slew_correction.time - calibrated_hits.time)[:n])
