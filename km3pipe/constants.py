# coding=utf-8
# Filename: constants.py
# pylint: disable=C0103
"""
The constants used in KM3Pipe.

"""
from __future__ import division, absolute_import, print_function

# TODO: this module should be refactored soon!

import math


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
