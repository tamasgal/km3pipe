# Filename: test_hardware.py
# pylint: disable=C0111,C0103,R0904
"""
Detector description (detx format v5)

global_det_id ndoms
dom_id line_id floor_id npmts
 pmt_id_global x y z dx dy dz t0
 pmt_id_global x y z dx dy dz t0
 ...
 pmt_id_global x y z dx dy dz t0
dom_id line_id floor_id npmts
 ...

"""

from copy import deepcopy
from os.path import join, dirname
from io import StringIO

import numpy as np

from km3pipe.testing import TestCase, data_path
from km3pipe.hardware import Detector, PMT
from km3pipe.math import qrot_yaw

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

EXAMPLE_DETX = StringIO(
    "\n".join(
        (
            "1 3",
            "1 1 1 3",
            " 1 1.1 1.2 1.3 -1.1  0.2  0.3 10",
            " 2 1.4 1.5 1.6  0.1 -1.2  0.3 20",
            " 3 1.7 1.8 1.9  0.1  0.2 -1.3 30",
            "2 1 2 3",
            " 4 2.1 2.2 2.3 -1.1  0.2  0.3 40",
            " 5 2.4 2.5 2.6  0.1 -1.2  0.3 50",
            " 6 2.7 2.8 2.9  0.1  0.2 -1.3 60",
            "3 1 3 3",
            " 7 3.1 3.2 3.3 -1.1  0.2  0.3 70",
            " 8 3.4 3.5 3.6  0.1 -1.2  0.3 80",
            " 9 3.7 3.8 3.9  0.1  0.2 -1.3 90",
        )
    )
)

EXAMPLE_DETX_MIXED_IDS = StringIO(
    "\n".join(
        (
            "1 3",
            "8 1 1 3",
            " 83 1.1 1.2 1.3 -1.1  0.2  0.3 10",
            " 81 1.4 1.5 1.6  0.1 -1.2  0.3 20",
            " 82 1.7 1.8 1.9  0.1  0.2 -1.3 30",
            "7 1 2 3",
            " 71 2.1 2.2 2.3 -1.1  0.2  0.3 40",
            " 73 2.4 2.5 2.6  0.1 -1.2  0.3 50",
            " 72 2.7 2.8 2.9  0.1  0.2 -1.3 60",
            "6 1 3 3",
            " 62 3.1 3.2 3.3 -1.1  0.2  0.3 70",
            " 63 3.4 3.5 3.6  0.1 -1.2  0.3 80",
            " 61 3.7 3.8 3.9  0.1  0.2 -1.3 90",
        )
    )
)

EXAMPLE_DETX_RADIAL = StringIO(
    "\n".join(
        (
            "1 3",
            "1 1 1 4",
            " 1 1 0 0 1 0 0 10",
            " 2 0 1 0 0 1 0 20",
            " 3 -1 0 0 -1 0 0 30",
            " 4 0 -1 0 0 -1 0 40",
            "2 1 2 2",
            " 5 0 0 1 0 0 1 50",
            " 6 0 0 -1 0 0 -1 60",
            "3 1 3 2",
            " 7 1 2 3 1 2 3 70",
            " 8 -3 -2 -1 -3 -2 -1 80",
            "4 2 1 2",
            " 9 0 0 1 0 0 1 90",
            " 10 0 0 -1 0 0 -1 100",
        )
    )
)


class TestDetector(TestCase):
    def setUp(self):
        self.det = Detector()
        self.det._det_file = EXAMPLE_DETX

    def test_parse_header_extracts_correct_det_id(self):
        self.det._parse_header()
        self.assertEqual(1, self.det.det_id)

    def test_parse_header_extracts_correct_n_doms(self):
        self.det._parse_header()
        self.assertEqual(3, self.det.n_doms)

    def test_parse_doms_maps_each_dom_correctly(self):
        self.det._parse_doms()
        expected = {1: (1, 1, 3), 2: (1, 2, 3), 3: (1, 3, 3)}
        self.assertDictEqual(expected, self.det.doms)

    def test_dom_ids(self):
        self.det._parse_doms()
        self.assertEqual((1, 2, 3), tuple(self.det.dom_ids))

    def test_parse_reset_cache(self):
        self.det._parse_doms()
        assert not self.det._dom_positions
        assert not self.det._pmt_angles
        assert not self.det._xy_positions
        self.det.dom_positions
        self.det.pmt_angles
        self.det.xy_positions
        assert self.det._dom_positions
        assert len(self.det._pmt_angles) == 3
        assert len(self.det._xy_positions) == 1
        self.det.reset_caches()
        assert not self.det._dom_positions
        assert not self.det._pmt_angles
        assert not self.det._xy_positions

    def test_parse_doms_maps_each_dom_correctly_for_mixed_pmt_ids(self):
        self.det._det_file = EXAMPLE_DETX_MIXED_IDS
        self.det._parse_doms()
        expected = {8: (1, 1, 3), 7: (1, 2, 3), 6: (1, 3, 3)}
        self.assertDictEqual(expected, self.det.doms)

    def test_dom_positions(self):
        self.det._parse_doms()
        assert np.allclose(
            [1.49992331, 1.51893187, 1.44185513], self.det.dom_positions[1]
        )
        assert np.allclose(
            [2.49992331, 2.51893187, 2.44185513], self.det.dom_positions[2]
        )
        assert np.allclose(
            [3.49992331, 3.51893187, 3.44185513], self.det.dom_positions[3]
        )

    def test_xy_positions(self):
        self.det._parse_doms()
        assert len(self.det.xy_positions) == 1
        assert np.allclose([1.49992331, 1.51893187], self.det.xy_positions[0])

    def test_correct_number_of_pmts(self):
        self.det._parse_doms()
        assert 9 == len(self.det.pmts)

    def test_pmt_attributes(self):
        self.det._parse_doms()
        assert (1, 2, 3, 4, 5, 6, 7, 8, 9) == tuple(self.det.pmts.pmt_id)
        assert np.allclose(
            [1.1, 1.4, 1.7, 2.1, 2.4, 2.7, 3.1, 3.4, 3.7], self.det.pmts.pos_x
        )
        assert np.allclose((1.7, 1.8, 1.9), self.det.pmts.pos[2])
        assert np.allclose((0.1, 0.2, -1.3), self.det.pmts.dir[8])

    def test_pmt_index_by_omkey(self):
        self.det._parse_doms()
        assert 5 == self.det._pmt_index_by_omkey[(1, 2, 2)]
        assert 0 == self.det._pmt_index_by_omkey[(1, 1, 0)]
        assert 4 == self.det._pmt_index_by_omkey[(1, 2, 1)]
        assert 1 == self.det._pmt_index_by_omkey[(1, 1, 1)]

    def test_pmt_index_by_pmt_id(self):
        self.det._parse_doms()
        assert 0 == self.det._pmt_index_by_pmt_id[1]

    def test_pmt_with_id_returns_correct_omkeys(self):
        self.det._parse_doms()
        pmt = self.det.pmt_with_id(1)
        assert (1, 1, 0) == (pmt.du, pmt.floor, pmt.channel_id)
        pmt = self.det.pmt_with_id(5)
        assert (1, 2, 1) == (pmt.du, pmt.floor, pmt.channel_id)

    def test_pmt_with_id_returns_correct_omkeys_with_mixed_pmt_ids(self):
        self.det._det_file = EXAMPLE_DETX_MIXED_IDS
        self.det._parse_doms()
        pmt = self.det.pmt_with_id(73)
        assert (1, 2, 1) == (pmt.du, pmt.floor, pmt.channel_id)
        pmt = self.det.pmt_with_id(81)
        assert (1, 1, 1) == (pmt.du, pmt.floor, pmt.channel_id)

    def test_pmt_with_id_raises_exception_for_invalid_id(self):
        self.det._parse_doms()
        with self.assertRaises(KeyError):
            self.det.pmt_with_id(100)

    def test_get_pmt(self):
        self.det._det_file = EXAMPLE_DETX_MIXED_IDS
        self.det._parse_doms()
        pmt = self.det.get_pmt(7, 2)
        assert (1, 2, 2) == (pmt.du, pmt.floor, pmt.channel_id)

    def test_xy_pos(self):
        self.det._parse_doms()
        xy = self.det.xy_positions
        assert xy is not None

    def test_ascii(self):
        detx_string = "\n".join(
            (
                "1 3",
                "1 1 1 3",
                " 1 1.1 1.2 1.3 1.1 2.1 3.1 10.0",
                " 2 1.4 1.5 1.6 4.1 5.1 6.1 20.0",
                " 3 1.7 1.8 1.9 7.1 8.1 9.1 30.0",
                "2 1 2 3",
                " 4 2.1 2.2 2.3 1.2 2.2 3.2 40.0",
                " 5 2.4 2.5 2.6 4.2 5.2 6.2 50.0",
                " 6 2.7 2.8 2.9 7.2 8.2 9.2 60.0",
                "3 1 3 3",
                " 7 3.1 3.2 3.3 1.3 2.3 3.3 70.0",
                " 8 3.4 3.5 3.6 4.3 5.3 6.3 80.0",
                " 9 3.7 3.8 3.9 7.3 8.3 9.3 90.0\n",
            )
        )
        detx_fob = StringIO(detx_string)

        self.det = Detector()
        self.det._det_file = detx_fob
        self.det._parse_header()
        self.det._parse_doms()
        assert detx_string == self.det.ascii

    def test_ascii_with_mixed_dom_ids(self):
        detx_string = "\n".join(
            (
                "1 3",
                "8 1 1 3",
                " 1 1.1 1.2 1.3 1.1 2.1 3.1 10.0",
                " 2 1.4 1.5 1.6 4.1 5.1 6.1 20.0",
                " 3 1.7 1.8 1.9 7.1 8.1 9.1 30.0",
                "4 1 2 3",
                " 4 2.1 2.2 2.3 1.2 2.2 3.2 40.0",
                " 5 2.4 2.5 2.6 4.2 5.2 6.2 50.0",
                " 6 2.7 2.8 2.9 7.2 8.2 9.2 60.0",
                "9 1 3 3",
                " 7 3.1 3.2 3.3 1.3 2.3 3.3 70.0",
                " 8 3.4 3.5 3.6 4.3 5.3 6.3 80.0",
                " 9 3.7 3.8 3.9 7.3 8.3 9.3 90.0\n",
            )
        )
        detx_fobj = StringIO(detx_string)

        self.det = Detector()
        self.det._det_file = detx_fobj
        self.det._parse_header()
        self.det._parse_doms()
        assert detx_string == self.det.ascii

    def test_init_from_string(self):
        detx_string = "\n".join(
            (
                "1 3",
                "8 1 1 3",
                " 1 1.1 1.2 1.3 1.1 2.1 3.1 10.0",
                " 2 1.4 1.5 1.6 4.1 5.1 6.1 20.0",
                " 3 1.7 1.8 1.9 7.1 8.1 9.1 30.0",
                "4 1 2 3",
                " 4 2.1 2.2 2.3 1.2 2.2 3.2 40.0",
                " 5 2.4 2.5 2.6 4.2 5.2 6.2 50.0",
                " 6 2.7 2.8 2.9 7.2 8.2 9.2 60.0",
                "9 1 3 3",
                " 7 3.1 3.2 3.3 1.3 2.3 3.3 70.0",
                " 8 3.4 3.5 3.6 4.3 5.3 6.3 80.0",
                " 9 3.7 3.8 3.9 7.3 8.3 9.3 90.0\n",
            )
        )
        det = Detector(string=detx_string)
        assert 1 == det.n_dus
        assert 3 == det.n_doms

    def test_detx_format_version_1(self):
        det = Detector(filename=data_path("detx/detx_v1.detx"))
        assert 2 == det.n_dus
        assert 6 == det.n_doms
        assert 3 == det.n_pmts_per_dom
        assert "v1" == det.version
        self.assertListEqual([1.1, 1.2, 1.3], list(det.pmts.pos[0]))
        self.assertListEqual([3.4, 3.5, 3.6], list(det.pmts.pos[7]))
        self.assertListEqual([23.4, 23.5, 23.6], list(det.pmts.pos[16]))

    def test_detx_v1_is_the_same_ascii(self):
        det = Detector(filename=data_path("detx/detx_v1.detx"))
        with open(data_path("detx/detx_v1.detx"), "r") as fobj:
            assert fobj.read() == det.ascii

    def test_detx_format_version_2(self):
        det = Detector(filename=data_path("detx/detx_v2.detx"))
        assert 2 == det.n_dus
        assert 6 == det.n_doms
        assert 3 == det.n_pmts_per_dom
        assert 256500.0 == det.utm_info.easting
        assert 4743000.0 == det.utm_info.northing
        assert "WGS84" == det.utm_info.ellipsoid
        assert "32N" == det.utm_info.grid
        assert -2425.0 == det.utm_info.z
        assert 1500000000.1 == det.valid_from
        assert 9999999999.0 == det.valid_until
        assert "v2" == det.version
        self.assertListEqual([1.1, 1.2, 1.3], list(det.pmts.pos[0]))
        self.assertListEqual([3.4, 3.5, 3.6], list(det.pmts.pos[7]))
        self.assertListEqual([23.4, 23.5, 23.6], list(det.pmts.pos[16]))

    def test_detx_v2_is_the_same_ascii(self):
        det = Detector(filename=data_path("detx/detx_v2.detx"))
        with open(data_path("detx/detx_v2.detx"), "r") as fobj:
            assert fobj.read() == det.ascii

    def test_detx_format_version_3(self):
        det = Detector(filename=data_path("detx/detx_v3.detx"))
        assert 2 == det.n_dus
        assert 6 == det.n_doms
        assert 3 == det.n_pmts_per_dom
        assert 256500.0 == det.utm_info.easting
        assert 4743000.0 == det.utm_info.northing
        assert "WGS84" == det.utm_info.ellipsoid
        assert "32N" == det.utm_info.grid
        assert -2425.0 == det.utm_info.z
        assert 1500000000.1 == det.valid_from
        assert 9999999999.0 == det.valid_until
        assert "v3" == det.version
        self.assertListEqual([1.1, 1.2, 1.3], list(det.pmts.pos[0]))
        self.assertListEqual([3.4, 3.5, 3.6], list(det.pmts.pos[7]))
        self.assertListEqual([23.4, 23.5, 23.6], list(det.pmts.pos[16]))

    def test_detector_repr(self):
        det = Detector(filename=data_path("detx/detx_v3.detx"))
        assert "Detector id: '23', n_doms: 6, dus: [1, 2]" == repr(det)

    def test_detx_format_version_3_with_whitespace(self):
        det = Detector(filename=data_path("detx/detx_v3_whitespace.detx"))
        assert 2 == det.n_dus
        assert 6 == det.n_doms
        assert 3 == det.n_pmts_per_dom
        assert 256500.0 == det.utm_info.easting
        assert 4743000.0 == det.utm_info.northing
        assert "WGS84" == det.utm_info.ellipsoid
        assert "32N" == det.utm_info.grid
        assert -2425.0 == det.utm_info.z
        assert 1500000000.1 == det.valid_from
        assert 9999999999.0 == det.valid_until
        assert "v3" == det.version
        self.assertListEqual([1.1, 1.2, 1.3], list(det.pmts.pos[0]))
        self.assertListEqual([3.4, 3.5, 3.6], list(det.pmts.pos[7]))
        self.assertListEqual([23.4, 23.5, 23.6], list(det.pmts.pos[16]))

    def test_detx_format_comments(self):
        det = Detector(filename=data_path("detx/detx_v1.detx"))
        assert len(det.comments) == 0

        det = Detector(filename=data_path("detx/detx_v2.detx"))
        assert len(det.comments) == 0

        det = Detector(filename=data_path("detx/detx_v3.detx"))
        assert len(det.comments) == 2
        assert " a comment line" == det.comments[0]
        assert " another comment line starting with '#'" == det.comments[1]

    def test_comments_are_written(self):
        det = Detector(filename=data_path("detx/detx_v3.detx"))
        det.add_comment("foo")
        assert 3 == len(det.comments)
        assert det.comments[2] == "foo"
        assert "# foo" == det.ascii.splitlines()[2]

    def test_detx_v3_is_the_same_ascii(self):
        det = Detector(filename=data_path("detx/detx_v3.detx"))
        with open(data_path("detx/detx_v3.detx"), "r") as fobj:
            assert fobj.read() == det.ascii

    def test_translate_detector(self):
        self.det._parse_doms()
        t_vec = np.array([1, 2, 3])
        orig_pos = self.det.pmts.pos.copy()
        self.det.translate_detector(t_vec)
        assert np.allclose(orig_pos + t_vec, self.det.pmts.pos)

    def test_translate_detector_updates_xy_positions(self):
        self.det._parse_doms()
        t_vec = np.array([1, 2, 3])
        orig_xy_pos = self.det.xy_positions.copy()
        self.det.translate_detector(t_vec)
        assert np.allclose(orig_xy_pos + t_vec[:2], self.det.xy_positions)

    def test_translate_detector_updates_dom_positions(self):
        self.det._parse_doms()
        t_vec = np.array([1, 2, 3])
        orig_dom_pos = deepcopy(self.det.dom_positions)
        self.det.translate_detector(t_vec)
        for dom_id, pos in self.det.dom_positions.items():
            assert np.allclose(orig_dom_pos[dom_id] + t_vec, pos)

    def test_rotate_dom_by_yaw(self):
        det = Detector()
        det._det_file = EXAMPLE_DETX_RADIAL
        det._parse_doms()
        # here, only one PMT is checked
        dom_id = 1
        heading = 23
        channel_id = 0
        pmt_dir = det.pmts[det.pmts.dom_id == dom_id].dir[channel_id].copy()
        pmt_dir_rot = qrot_yaw(pmt_dir, heading)
        det.rotate_dom_by_yaw(dom_id, heading)
        assert np.allclose(
            pmt_dir_rot, det.pmts[det.pmts.dom_id == dom_id].dir[channel_id]
        )
        assert np.allclose(
            [0.92050485, 0.39073113, 0],
            det.pmts[det.pmts.dom_id == dom_id].pos[channel_id],
        )

    def test_rotate_dom_set_by_step_by_360_degrees(self):
        det = Detector()
        det._det_file = EXAMPLE_DETX_RADIAL
        det._parse_doms()
        dom_id = 1
        channel_id = 0
        pmt_dir = det.pmts[det.pmts.dom_id == dom_id].dir[channel_id].copy()
        pmt_pos = det.pmts[det.pmts.dom_id == dom_id].pos[channel_id].copy()
        for i in range(36):
            det.rotate_dom_by_yaw(dom_id, 10)
        pmt_dir_rot = det.pmts[det.pmts.dom_id == dom_id].dir[channel_id]
        assert np.allclose(pmt_dir, pmt_dir_rot)
        pmt_pos_rot = det.pmts[det.pmts.dom_id == dom_id].pos[channel_id]
        assert np.allclose(pmt_pos, pmt_pos_rot)

    def test_rotate_du_by_yaw_step_by_step_360_degrees(self):
        det = Detector()
        det._det_file = EXAMPLE_DETX_RADIAL
        det._parse_doms()
        du = 2
        pmt_dir = det.pmts[det.pmts.du == du].dir.copy()
        pmt_pos = det.pmts[det.pmts.du == du].pos.copy()
        pmt_dir_other_dus = det.pmts[det.pmts.du != du].dir.copy()
        pmt_pos_other_dus = det.pmts[det.pmts.du != du].pos.copy()
        for i in range(36):
            det.rotate_du_by_yaw(du, 10)
        pmt_dir_rot = det.pmts[det.pmts.du == du].dir
        pmt_pos_rot = det.pmts[det.pmts.du == du].pos
        assert np.allclose(pmt_dir, pmt_dir_rot)
        assert np.allclose(pmt_pos, pmt_pos_rot)
        assert np.allclose(pmt_dir_other_dus, det.pmts[det.pmts.du != du].dir)
        assert np.allclose(pmt_pos_other_dus, det.pmts[det.pmts.du != du].pos)

    def test_rescale_detector(self):
        self.det._parse_doms()
        dom_positions = deepcopy(self.det.dom_positions)
        scale_factor = 2
        self.det.rescale(scale_factor)
        for dom_id, dom_pos in self.det.dom_positions.items():
            assert np.allclose(dom_pos, dom_positions[dom_id] * scale_factor)

    def test_dom_table(self):
        self.det._parse_doms()
        dt = self.det.dom_table
        assert 3 == len(dt)
        assert np.allclose([1, 2, 3], dt.dom_id)
        assert np.allclose([1, 1, 1], dt.du)
        assert np.allclose([1, 2, 3], dt.floor)
        assert np.allclose([1.49992331, 2.49992331, 3.49992331], dt.pos_x)
        assert np.allclose([1.51893187, 2.51893187, 3.51893187], dt.pos_y)
        assert np.allclose([1.44185513, 2.44185513, 3.44185513], dt.pos_z)

    def test_dom_table_with_another_detx(self):
        det = Detector()
        det._det_file = EXAMPLE_DETX_RADIAL
        det._parse_doms()

        dt = det.dom_table
        assert 4 == len(dt)
        assert np.allclose([1, 2, 3, 4], dt.dom_id)
        assert np.allclose([1, 1, 1, 2], dt.du)
        assert np.allclose([1, 2, 3, 1], dt.floor)
        assert np.allclose([0, 0, 0, 0], dt.pos_x)
        assert np.allclose([0, 0, 0, 0], dt.pos_y)
        assert np.allclose([0, 0, 0, 0], dt.pos_z)

    def test_center_of_mass(self):
        det = Detector()
        det._det_file = EXAMPLE_DETX
        det._parse_doms()

        assert np.allclose([2.4, 2.5, 2.6], det.com)

    def test_center_of_mass_with_another_detx(self):
        det = Detector()
        det._det_file = EXAMPLE_DETX_RADIAL
        det._parse_doms()

        assert np.allclose([-0.2, 0.0, 0.2], det.com)

    def test_jdetectordb_output_with_detx_v3(self):
        det = Detector(
            data_path("detx/D_ORCA006_t.A02181836.p.A02181837.r.A02182001.detx")
        )
        assert det.utm_info is not None
        assert det.utm_info.ellipsoid == "WGS84"
        assert det.utm_info.grid == "32N"
        assert det.utm_info.easting == 256500.0
        assert det.utm_info.northing == 4743000.0
        assert det.utm_info.z == -2440.0
