# Filename: test_physics.py
# pylint: disable=locally-disabled

import math
import numpy as np
import pandas as pd

from km3pipe.dataclasses import Table
from km3pipe.testing import TestCase, data_path
from km3pipe.physics import get_cherenkov
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
            "pos_x": 0.0,
            "pos_y": 0.0,
            "pos_z": 0.0,
            "dir_x": 0.0,
            "dir_y": 1.0,
            "dir_z": 0.0,
            "t": 0.0,
        }
        self.calib_hits = {
            "pos_x": [1.0, 1.0],
            "pos_y": [0.0, 0.0],
            "pos_z": [0.0, 0.0],
            "dir_x": [0.0, 0.0],
            "dir_y": [0.0, 0.0],
            "dir_z": [0.0, 0.0],
        }

        self.arr_track = np.array(
            [(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)],
            dtype=[
                ("pos_x", "<f8"),
                ("pos_y", "<f8"),
                ("pos_z", "<f8"),
                ("dir_x", "<f8"),
                ("dir_y", "<f8"),
                ("dir_z", "<f8"),
                ("t", "<f8"),
            ],
        )

        self.arr_calib_hits = np.array(
            [(1.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
            dtype=[
                ("pos_x", "<f8"),
                ("pos_y", "<f8"),
                ("pos_z", "<f8"),
                ("dir_x", "<f8"),
                ("dir_y", "<f8"),
                ("dir_z", "<f8"),
            ],
        )

    def test_with_basic_values(self):
        trks = [
            self.track,
            pd.Series(self.track),
        ]  # , Table(self.track), self.arr_track]
        calib_hits = [
            self.calib_hits,
            pd.DataFrame(self.calib_hits),
        ]  # , Table(self.calib_hits), self.arr_calib_hits]

        for trk, hits in zip(trks, calib_hits):

            arr = get_cherenkov(hits, trk)

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
