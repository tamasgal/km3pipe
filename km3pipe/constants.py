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
WATER_INDEX = 1.3499
DN_DL = 0.0298
COS_CHERENKOV = 1 / WATER_INDEX
CHERENKOV_ANGLE_RAD = math.acos(COS_CHERENKOV)
SIN_CHERENKOV = math.sin(CHERENKOV_ANGLE_RAD)
TAN_CHERENKOV = math.tan(CHERENKOV_ANGLE_RAD)
C_LIGHT = 299792458e-9
V_LIGHT_WATER = C_LIGHT / (WATER_INDEX + DN_DL)

# Detector related parameters
arca_frame_duration = 0.1  # s
orca_frame_duration = 0.1  # s

c = 2.99792458e8  # m/s

n_water_antares_phase = 1.3499
n_water_antares_group = 1.3797
n_water_km3net_group = 1.3787
n_water_antares = n_water_antares_group
theta_cherenkov_water_antares = math.acos(1 / n_water_antares_phase)
theta_cherenkov_water_km3net = math.acos(1 / n_water_km3net_group)
c_water_antares = c / n_water_antares_group
c_water_km3net = c / n_water_km3net_group

# Math
pi = math.pi
e = math.e

# Default values for time residuals
dt_window_l = -15  # ns
dt_window_h = +25  # ns
