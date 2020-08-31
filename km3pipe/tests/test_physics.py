# Filename: test_physics.py
# pylint: disable=locally-disabled

import math
import pandas as pd

from km3pipe.testing import TestCase, data_path
from km3pipe.physics import get_cherenkov_photon
from km3pipe.constants import SIN_CHERENKOV, TAN_CHERENKOV, V_LIGHT_WATER, C_LIGHT

__author__ = "Zineb ALY"
__copyright__ = "Copyright 2017, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Zineb ALY"
__email__ = "zaly@km3net.de"
__status__ = "Development"


class TestGetCherenkovPhotons(TestCase):
    def setUp(self):
        self.expected_t = -1 / (TAN_CHERENKOV * C_LIGHT) + 1 / (
            SIN_CHERENKOV * V_LIGHT_WATER
        )
        self.track = {
            "pos_x": 0,
            "pos_y": 0,
            "pos_z": 0,
            "dir_x": 0,
            "dir_y": 1,
            "dir_z": 0,
            "t": 0,
        }
        self.calib_hits = {
            "pos_x": [1, 1],
            "pos_y": [0, 0],
            "pos_z": [0, 0],
            "dir_x": [0, 0],
            "dir_y": [0, 0],
            "dir_z": [0, 0],
        }

    def test_with_basic_values(self):
        trk = pd.Series(self.track)
        hits = pd.DataFrame(self.calib_hits)

        arr = get_cherenkov_photon(hits, trk)

        assert arr["d_photon_closest"][0] == 1
        assert arr["d_photon_closest"][1] == 1

        assert arr["d_photon"][0] == 1 / SIN_CHERENKOV
        assert arr["d_photon"][1] == 1 / SIN_CHERENKOV

        assert arr["d_track"][0] == -1 / TAN_CHERENKOV
        assert arr["d_track"][1] == -1 / TAN_CHERENKOV

        assert arr["t_photon"][0] == self.expected_t
        assert arr["t_photon"][1] == self.expected_t

        assert arr["cos_photon_PMT"][0] == 0
        assert arr["cos_photon_PMT"][1] == 0

        assert arr["dir_z_photon"][0] == 0
        assert arr["dir_z_photon"][1] == 0
