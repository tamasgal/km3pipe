# coding=utf-8
# Filename: astro.py
# pylint: disable=C0103
# pragma: no cover
"""
Astro utils.
"""
from __future__ import division, absolute_import, print_function


from astropy import units as u
from astropy.units import degree, minute
from astropy.coordinates import (Angle, Latitude, Longitude, Galactic,
                                 EarthLocation, AltAz, SkyCoord)
from astropy.time import Time

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


def to_frame(times, zeniths, azimuths, frame='galactic'):
    zeniths *= degree
    azimuths *= degree
    orca_frame = AltAz(location=orca_loc, obstime=times)
    coords = SkyCoord(zeniths, azimuths, frame=orca_frame)
    return coords
