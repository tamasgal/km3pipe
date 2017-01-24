# coding=utf-8
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
from __future__ import division, absolute_import, print_function

from km3pipe.testing import TestCase, StringIO, skipIf
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

EXAMPLE_DETX_WRITE = StringIO("\n".join((
    "1 3",
    "1 1 1 3",
    " 1 1.1 1.2 1.3 1.0 0.0 0.0 10.0",
    " 2 1.4 1.5 1.6 0.0 1.0 0.0 20.0",
    " 3 1.7 1.8 1.9 0.0 0.0 1.0 30.0",
    "2 1 2 3",
    " 4 2.1 2.2 2.3 0.0 1.0 0.0 40.0",
    " 5 2.4 2.5 2.6 0.0 0.0 1.0 50.0",
    " 6 2.7 2.8 2.9 1.0 0.0 0.0 60.0",
    "3 1 3 3",
    " 7 3.1 3.2 3.3 0.0 0.0 1.0 70.0",
    " 8 3.4 3.5 3.6 0.0 1.0 0.0 80.0",
    " 9 3.7 3.8 3.9 1.0 0.0 0.0 90.0\n",)))

EXAMPLE_MC_DETX_WRITE_MIXED_IDS = StringIO("\n".join((
    "-1 3",
    "6 1 1 3",
    " 31 1.1 1.2 1.3 1.0 0.0 0.0 10.0",
    " 22 1.4 1.5 1.6 0.0 1.0 0.0 20.0",
    " 13 1.7 1.8 1.9 0.0 0.0 1.0 30.0",
    "3 1 2 3",
    " 34 2.1 2.2 2.3 1.0 0.0 0.0 40.0",
    " 45 2.4 2.5 2.6 0.0 1.0 0.0 50.0",
    " 16 2.7 2.8 2.9 0.0 0.0 1.0 60.0",
    "9 1 3 3",
    " 17 3.1 3.2 3.3 1.0 0.0 0.0 70.0",
    " 48 3.4 3.5 3.6 0.0 1.0 0.0 80.0",
    " 39 3.7 3.8 3.9 0.0 0.0 1.0 90.0\n",)))


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

    def test_parse_doms_maps_each_dom_correctly_for_mixed_pmt_ids(self):
        self.det._det_file = EXAMPLE_DETX_MIXED_IDS
        self.det._parse_doms()
        expected = {8: (1, 1, 3), 7: (1, 2, 3), 6: (1, 3, 3)}
        self.assertDictEqual(expected, self.det.doms)

    def test_dom_positions(self):
        self.det._parse_doms()
        self.assertAlmostEqual(1.4, self.det.dom_positions[1][0])
        self.assertAlmostEqual(1.5, self.det.dom_positions[1][1])
        self.assertAlmostEqual(1.6, self.det.dom_positions[1][2])
        self.assertAlmostEqual(2.4, self.det.dom_positions[2][0])
        self.assertAlmostEqual(2.5, self.det.dom_positions[2][1])
        self.assertAlmostEqual(2.6, self.det.dom_positions[2][2])

    def test_pmt_with_id_returns_correct_omkeys(self):
        self.det._parse_doms()
        self.assertEqual((1, 1, 0), self.det.pmt_with_id(1).omkey)
        self.assertEqual((1, 2, 1), self.det.pmt_with_id(5).omkey)

    def test_pmt_with_id_returns_correct_omkeys_with_mixed_pmt_ids(self):
        self.det._det_file = EXAMPLE_DETX_MIXED_IDS
        self.det._parse_doms()
        self.assertEqual((1, 2, 1), self.det.pmt_with_id(73).omkey)
        self.assertEqual((1, 1, 1), self.det.pmt_with_id(81).omkey)

    def test_pmt_with_id_raises_exception_for_invalid_id(self):
        self.det._parse_doms()
        with self.assertRaises(KeyError):
            self.det.pmt_with_id(100)

    def test_get_pmt(self):
        self.det._det_file = EXAMPLE_DETX_MIXED_IDS
        self.det._parse_doms()
        pmt = self.det.get_pmt(7, 2)
        self.assertEqual((1, 2, 2), pmt.omkey)

    def test_pmtid2omkey_old(self):
        pmtid2omkey = self.det._pmtid2omkey_old
        self.assertEqual((1, 13, 12), tuple(pmtid2omkey(168)))
        self.assertEqual((1, 12, 18), tuple(pmtid2omkey(205)))
        self.assertEqual((1, 11, 22), tuple(pmtid2omkey(240)))
        self.assertEqual((4, 11, 2), tuple(pmtid2omkey(1894)))
        self.assertEqual((9, 18, 0), tuple(pmtid2omkey(4465)))
        self.assertEqual((95, 7, 16), tuple(pmtid2omkey(52810)))
        self.assertEqual((95, 4, 13), tuple(pmtid2omkey(52900)))

    def test_pmtid2omkey_old_handles_floats(self):
        pmtid2omkey = self.det._pmtid2omkey_old
        self.assertEqual((1, 13, 12), tuple(pmtid2omkey(168.0)))
        self.assertEqual((1, 12, 18), tuple(pmtid2omkey(205.0)))
        self.assertEqual((1, 11, 22), tuple(pmtid2omkey(240.0)))
        self.assertEqual((4, 11, 2), tuple(pmtid2omkey(1894.0)))
        self.assertEqual((9, 18, 0), tuple(pmtid2omkey(4465.0)))
        self.assertEqual((95, 7, 16), tuple(pmtid2omkey(52810.0)))
        self.assertEqual((95, 4, 13), tuple(pmtid2omkey(52900.0)))


class TestPMT(TestCase):

    def test_init(self):
        pmt = PMT(1, (1., 2, 3), (4., 5, 6), 7, 8, (9, 10, 11))
        self.assertAlmostEqual(1, pmt.id)
        self.assertAlmostEqual(1, pmt.pos.x)
        self.assertAlmostEqual(2, pmt.pos.y)
        self.assertAlmostEqual(3, pmt.pos.z)
        self.assertAlmostEqual(4, pmt.dir.x, 6)
        self.assertAlmostEqual(5, pmt.dir.y, 6)
        self.assertAlmostEqual(6, pmt.dir.z, 6)
        self.assertAlmostEqual(7, pmt.t0)
        self.assertAlmostEqual(8, pmt.channel_id)
        self.assertEqual((9, 10, 11), pmt.omkey)
