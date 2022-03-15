# Filename: constants.py
# pylint: disable=C0103
# pragma: no cover
"""
The constants used in KM3Pipe.

"""
import math

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

# Physics constants used in physics module
WATER_INDEX = 1.3499  # Used in aanet
INDEX_OF_REFRACTION_WATER = 1.3800851282  # Used in Jpp (e.g. in PDFs)
DN_DL = 0.0298
COS_CHERENKOV = 1 / WATER_INDEX
CHERENKOV_ANGLE_RAD = math.acos(COS_CHERENKOV)
SIN_CHERENKOV = math.sin(CHERENKOV_ANGLE_RAD)
TAN_CHERENKOV = math.tan(CHERENKOV_ANGLE_RAD)
C_LIGHT = 299792458e-9  # m/ns
V_LIGHT_WATER = C_LIGHT / (WATER_INDEX + DN_DL)
C_WATER = C_LIGHT / INDEX_OF_REFRACTION_WATER

c = 2.99792458e8  # m/s
