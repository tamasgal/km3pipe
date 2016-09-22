#!/usr/bin/env python
"""
Orca Astro Test.

You are in a rubber boat floating atop ORCA. If a neutrino is coming at you
from the galactic center, should you protect yourself with a helmet, a vest
or with leaden boots?
"""

from astropy.coordinates import (Angle, Latitude, Longitude, Galactic, # noqa
                                 EarthLocation, AltAz, SkyCoord)  # noqa
from astropy.units import degree, minute, meter     # noqa
from astropy.time import Time
import matplotlib.pyplot as plt

from km3pipe.constants import orca_longitude, orca_latitude, orca_height


orca_loc = EarthLocation.from_geodetic(
    Longitude(orca_longitude * degree),
    Latitude(orca_latitude * degree),
    height=orca_height
)
orca_frame = AltAz(obstime=Time.now(), location=orca_loc)

gc_evt = SkyCoord(180 * degree, 0 * degree, frame='galactic')
evt = gc_evt.transform_to(orca_frame)

plt.subplot(111, projection='mollweide')
plt.plot(evt.az.deg, evt.alt.deg, 'x', markersize=5)
print(evt)
