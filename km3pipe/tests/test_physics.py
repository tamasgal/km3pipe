# Filename: test_physics.py
# pylint: disable=locally-disabled

import km3io as ki
import numpy as np

from km3pipe.dataclasses import Table
from km3pipe.testing import TestCase, data_path
from km3pipe.hardware import Detector
from km3pipe.calib import Calibration
from km3pipe.physics import cherenkov, get_closest, cut4d
import km3pipe.extras


__author__ = "Zineb ALY"
__copyright__ = "Copyright 2017, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Zineb ALY"
__email__ = "zaly@km3net.de"
__status__ = "Development"


class TestGetCherenkov(TestCase):
    def setUp(self):
        self.calib_hits = {
            "pos_x": [
                19.522,
                19.435,
                19.496,
                19.544,
                9.48,
                9.644,
                9.441,
                9.739,
                9.714,
                9.72,
            ],
            "pos_y": [
                -12.053,
                -12.194,
                -12.196,
                -12.289,
                6.013,
                6.086,
                5.852,
                5.769,
                6.02,
                5.786,
            ],
            "pos_z": [
                76.662,
                76.662,
                76.381,
                76.381,
                116.249,
                116.079,
                116.249,
                66.762,
                66.71,
                66.932,
            ],
            "dir_x": [
                -0.39,
                -0.831,
                -0.527,
                -0.279,
                -0.606,
                0.22,
                -0.797,
                0.696,
                0.57,
                0.606,
            ],
            "dir_y": [
                0.735,
                0.03,
                0.018,
                -0.447,
                0.57,
                0.93,
                -0.239,
                -0.655,
                0.606,
                -0.57,
            ],
            "dir_z": [
                0.555,
                0.555,
                -0.85,
                -0.85,
                0.555,
                -0.295,
                0.555,
                -0.295,
                -0.555,
                0.555,
            ],
        }

        self.track = {
            "pos_x": 1.7524502152598151,
            "pos_y": 39.06202405657308,
            "pos_z": 130.44049806891948,
            "dir_x": 0.028617421257374293,
            "dir_y": -0.489704257367248,
            "dir_z": -0.8714188335794505,
            "t": 70311441.452294,
        }

    def test_cherenkov_from_dict(self):

        arr = cherenkov(self.calib_hits, self.track)

        self.assertAlmostEqual(arr["d_photon_closest"][0], 24.049593557846112)
        self.assertAlmostEqual(arr["d_photon_closest"][1], 24.085065395206847)

        self.assertAlmostEqual(arr["d_photon"][0], 35.80244420413484)
        self.assertAlmostEqual(arr["d_photon"][1], 35.855250854478896)

        self.assertAlmostEqual(arr["d_track"][0], 45.88106599210481)
        self.assertAlmostEqual(arr["d_track"][1], 45.90850564175342)

        self.assertAlmostEqual(arr["t_photon"][0], 70311759.26448613)
        self.assertAlmostEqual(arr["t_photon"][1], 70311759.59904088)

        self.assertAlmostEqual(arr["cos_photon_PMT"][0], -0.98123942583677)
        self.assertAlmostEqual(arr["cos_photon_PMT"][1], -0.6166369315726149)

        self.assertAlmostEqual(arr["dir_x_photon"][0], 0.45964884122649263)
        self.assertAlmostEqual(arr["dir_x_photon"][1], 0.45652355929477095)

        self.assertAlmostEqual(arr["dir_y_photon"][0], -0.8001372907490844)
        self.assertAlmostEqual(arr["dir_y_photon"][1], -0.8025165828910586)

        self.assertAlmostEqual(arr["dir_z_photon"][0], -0.3853612055096594)
        self.assertAlmostEqual(arr["dir_z_photon"][1], -0.38412676812960095)

    def test_cherenkov_from_Table(self):

        arr = cherenkov(Table(self.calib_hits), Table(self.track))

        self.assertAlmostEqual(arr["d_photon_closest"][0], 24.049593557846112)
        self.assertAlmostEqual(arr["d_photon"][0], 35.80244420413484)
        self.assertAlmostEqual(arr["d_track"][0], 45.88106599210481)
        self.assertAlmostEqual(arr["t_photon"][0], 70311759.26448613)
        self.assertAlmostEqual(arr["cos_photon_PMT"][0], -0.98123942583677)
        self.assertAlmostEqual(arr["dir_x_photon"][0], 0.45964884122649263)
        self.assertAlmostEqual(arr["dir_y_photon"][0], -0.8001372907490844)
        self.assertAlmostEqual(arr["dir_z_photon"][0], -0.3853612055096594)

    def test_cherenkov_from_DataFrame(self):

        pd = km3pipe.extras.pandas()

        arr = cherenkov(pd.DataFrame(self.calib_hits), pd.Series(self.track))

        self.assertAlmostEqual(arr["d_photon_closest"][0], 24.049593557846112)
        self.assertAlmostEqual(arr["d_photon"][0], 35.80244420413484)
        self.assertAlmostEqual(arr["d_track"][0], 45.88106599210481)
        self.assertAlmostEqual(arr["t_photon"][0], 70311759.26448613)
        self.assertAlmostEqual(arr["cos_photon_PMT"][0], -0.98123942583677)
        self.assertAlmostEqual(arr["dir_x_photon"][0], 0.45964884122649263)
        self.assertAlmostEqual(arr["dir_y_photon"][0], -0.8001372907490844)
        self.assertAlmostEqual(arr["dir_z_photon"][0], -0.3853612055096594)


class TestGetClosest(TestCase):
    def setUp(self):
        self.track = {
            "pos_x": 1.7524502152598151,
            "pos_y": 39.06202405657308,
            "pos_z": 130.44049806891948,
            "dir_x": 0.028617421257374293,
            "dir_y": -0.489704257367248,
            "dir_z": -0.8714188335794505,
        }

        self.det = Detector(data_path("detx/detx_v3.detx"))

        pd = km3pipe.extras.pandas()

        self.DU = pd.DataFrame(self.det.dom_table).mean()

    def test_get_closest(self):
        pd = km3pipe.extras.pandas()

        DU = pd.DataFrame(self.det.dom_table).mean()
        d_closest, z_closest = get_closest(self.track, DU)

        self.assertAlmostEqual(d_closest, 9.073491762564467)
        self.assertAlmostEqual(z_closest, 82.24928115091757)

    def test_get_closest_from_DataFrame(self):
        pd = km3pipe.extras.pandas()
        d_closest, z_closest = get_closest(pd.Series(self.track), pd.Series(self.DU))

        self.assertAlmostEqual(d_closest, 9.073491762564467)
        self.assertAlmostEqual(z_closest, 82.24928115091757)

    def test_get_closest_from_Table(self):
        trk = {
            "pos_x": 1.7524502152598151,
            "pos_y": 39.06202405657308,
            "pos_z": 130.44049806891948,
            "dir_x": 0.028617421257374293,
            "dir_y": -0.489704257367248,
            "dir_z": -0.8714188335794505,
        }

        pd = km3pipe.extras.pandas()
        mean = pd.DataFrame(self.det.dom_table).mean()

        d_closest, z_closest = get_closest(Table(trk), mean)

        self.assertAlmostEqual(d_closest, 9.073491762564467)
        self.assertAlmostEqual(z_closest, 82.24928115091757)


class TestCut4D(TestCase):
    def test_cut4d(self):
        point4d = Table({"pos_x": [0], "pos_y": [3], "pos_z": [0], "t": [20]})

        items = Table(
            {
                "pos_x": [0, 10, 0, 20, 0],
                "pos_y": [10, 0, 0, 0, 30],
                "pos_z": [0, 0, 10, 0, 0],
                "time": [60, 15, 40, 20, 100],
            }
        )

        tmin = -50.0
        tmax = 10.0
        rmin = 3.0
        rmax = 80.0

        selected_items = cut4d(point4d, tmin, tmax, rmin, rmax, items)
        assert len(selected_items) == 3

        self.assertListEqual(
            list(np.array([0, 10, 0, 60])),
            list(selected_items.T[0]),
        )
        self.assertListEqual(
            list(np.array([0, 0, 10, 40])),
            list(selected_items.T[1]),
        )
        self.assertListEqual(
            list(np.array([0, 30, 0, 100])),
            list(selected_items.T[2]),
        )
