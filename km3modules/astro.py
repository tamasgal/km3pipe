# coding=utf-8
# Filename: astro.py
# pylint: disable=C0103
# pragma: no cover
"""
Astro utils.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from astropy.units import degree, rad
from astropy.coordinates import (Latitude, Longitude,
                                 EarthLocation, AltAz, SkyCoord)

from km3pipe.constants import orca_longitude, orca_latitude, orca_height

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"


orca_loc = EarthLocation.from_geodetic(
    Longitude(orca_longitude * degree),
    Latitude(orca_latitude * degree),
    height=orca_height
)


def to_frame(time, zenith, azimuth, frame='galactic', unit=rad):
    """Tranform from the Detector frame to anything else.

    Parameters:
    -----------
    frame: str [default: 'galactic']
        The reference frame to transform to (needs to be known to astropy)

    """
    altitude = (0.5*np.pi - zenith) * unit
    azimuth *= unit
    orca_frame = AltAz(location=orca_loc, obstime=time)
    coords = SkyCoord(altitude, azimuth, frame=orca_frame)
    return coords.transform_to(frame)
