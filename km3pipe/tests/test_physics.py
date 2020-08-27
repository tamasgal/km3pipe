# Filename: test_physics.py
# pylint: disable=locally-disabled

import math
import pandas as pd

from km3pipe.testing import TestCase, data_path
from km3pipe.physics import get_cherenkov_photon

__author__ = "Zineb ALY"
__copyright__ = "Copyright 2017, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Zineb ALY"
__email__ = "zaly@km3net.de"
__status__ = "Development"


# Physics constants
WATER_INDEX = 1.3499
DN_DL = 0.0298
COS_CHERENKOV = 1 / WATER_INDEX
CHERENKOV_ANGLE_RAD = math.acos(COS_CHERENKOV)
SIN_CHERENKOV = math.sin(CHERENKOV_ANGLE_RAD)
TAN_CHERENKOV = math.tan(CHERENKOV_ANGLE_RAD)
C_LIGHT = 299792458e-9
V_LIGHT_WATER = C_LIGHT / (WATER_INDEX + DN_DL)


class TestGetCherenkovPhotons(TestCase):
    def test_with_basic_values(self):

        track = pd.Series(
            {
                "pos_x": 0,
                "pos_y": 0,
                "pos_z": 0,
                "dir_x": 0,
                "dir_y": 1,
                "dir_z": 0,
                "t": 0,
            }
        )
        calib_hits = pd.DataFrame(
            {
                "pos_x": [1, 1],
                "pos_y": [0, 0],
                "pos_z": [0, 0],
                "dir_x": [0, 0],
                "dir_y": [0, 0],
                "dir_z": [0, 0],
            }
        )

        df = get_cherenkov_photon(calib_hits, track)

        assert df["d_photon_closest"].iloc[0] == 1
        assert df["d_photon_closest"].iloc[1] == 1

        assert df["d_photon"].iloc[0] == 1 / SIN_CHERENKOV
        assert df["d_photon"].iloc[1] == 1 / SIN_CHERENKOV

        assert df["d_track"].iloc[0] == -1 / TAN_CHERENKOV
        assert df["d_track"].iloc[1] == -1 / TAN_CHERENKOV

        t = -1 / (TAN_CHERENKOV * C_LIGHT) + 1 / (SIN_CHERENKOV * V_LIGHT_WATER)

        assert df["t_photon"].iloc[0] == t
        assert df["t_photon"].iloc[1] == t

        assert df["cos_photon_PMT"].iloc[0] == 0
        assert df["cos_photon_PMT"].iloc[1] == 0

        assert df["dir_z_photon"].iloc[0] == 0
        assert df["dir_z_photon"].iloc[1] == 0
