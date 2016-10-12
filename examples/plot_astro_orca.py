#!/usr/bin/env python
"""
================
Orca Astro Test.
================

This example needs Astropy: `pip install astropy`.

Take some events in the detector and transform them
to galactic coordinates.
"""

from astropy.coordinates import (Angle, Latitude, Longitude, Galactic, # noqa
                                 EarthLocation, AltAz, SkyCoord)  # noqa
from astropy.units import degree, minute, meter     # noqa
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np

from km3pipe.constants import orca_longitude, orca_latitude, orca_height
import km3pipe.style


# load orca coordinates
orca_loc = EarthLocation.from_geodetic(
    Longitude(orca_longitude * degree),
    Latitude(orca_latitude * degree),
    height=orca_height
)
orca_frame = AltAz(obstime=Time.now(), location=orca_loc)

# prepare canvases
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111, projection='aitoff')

# generate some zenith + azimuth pairs
az = np.random.rand(100)*360.0 * degree
alt = (np.random.rand(100)*180.0-90.0) * degree

orca_event = SkyCoord(alt=alt, az=az, frame=orca_frame)
orca_event_origin = orca_event.galactic
ax.plot(orca_event_origin.l, orca_event_origin.b, 'o',
        markersize=10, alpha=0.3, color='k')
plt.show()
