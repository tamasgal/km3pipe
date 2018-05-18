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
from io import StringIO

import numpy as np

from km3pipe.testing import TestCase
from km3pipe.hardware import Detector, PMT

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


EXAMPLE_DETX = StringIO("\n".join((
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
    " 9 3.7 3.8 3.9  0.1  0.2 -1.3 90",)))

EXAMPLE_DETX_MIXED_IDS = StringIO("\n".join((
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
    " 61 3.7 3.8 3.9  0.1  0.2 -1.3 90",)))


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
        assert np.allclose([1.49992331, 1.51893187, 1.44185513],
                           self.det.dom_positions[1])
        assert np.allclose([2.49992331, 2.51893187, 2.44185513],
                           self.det.dom_positions[2])
        assert np.allclose([3.49992331, 3.51893187, 3.44185513],
                           self.det.dom_positions[3])

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
        assert np.allclose([1.1, 1.4, 1.7, 2.1, 2.4, 2.7, 3.1, 3.4, 3.7],
                           self.det.pmts.pos_x)
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
        detx_string = "\n".join((
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
            " 9 3.7 3.8 3.9 7.3 8.3 9.3 90.0\n",))
        detx_fob = StringIO(detx_string)

        self.det = Detector()
        self.det._det_file = detx_fob
        self.det._parse_header()
        self.det._parse_doms()
        assert detx_string == self.det.ascii

    def test_ascii_with_mixed_dom_ids(self):
        detx_string = "\n".join((
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
            " 9 3.7 3.8 3.9 7.3 8.3 9.3 90.0\n",))
        detx_fobj = StringIO(detx_string)

        self.det = Detector()
        self.det._det_file = detx_fobj
        self.det._parse_header()
        self.det._parse_doms()
        assert detx_string == self.det.ascii

    def test_translate_detector(self):
        self.det._parse_doms()
        t_vec = np.array([1, 2, 3])
        orig_pos = self.det.pmts.pos.copy()
        self.det.translate_detector(t_vec)
        assert np.allclose(orig_pos + t_vec, self.det.pmts.pos)
