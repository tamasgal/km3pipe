# coding=utf-8                    
# Filename: test_hardware.py
# pylint: disable=locally-disabled,C0111
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

from km3pipe.testing import *
from km3pipe.hardware import Detector, PMT

example_detx = StringIO("\n".join((
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

class TestDetector(TestCase):

    def setUp(self):
        self.det = Detector()
        self.det.det_file = example_detx

    def test_parse_header_extracts_correct_det_id(self):
        self.det.parse_header()
        self.assertEqual(1, self.det.det_id)

    def test_parse_header_extracts_correct_n_doms(self):
        self.det.parse_header()
        self.assertEqual(3, self.det.n_doms)

    def test_parse_doms_maps_each_dom_correctly(self):
        self.det.parse_doms()
        expected = {1: (1, 1, 3), 2: (1, 2, 3), 3: (1, 3, 3)}
        self.assertDictEqual(expected, self.det.doms)

    def test_parse_doms_fills_pmts_dict(self):
        self.det.parse_doms()
        self.assertEqual(9, len(self.det.pmts))
        self.assertTupleEqual((7, 3.1, 3.2, 3.3, -1.1,  0.2,  0.3, 70),
                              self.det.pmts[(1, 3, 0)])

    def test_pmtid2omkey(self):
        pmtid2omkey = self.det.pmtid2omkey
        self.assertEqual((1, 13, 12), tuple(pmtid2omkey(168)))
        self.assertEqual((1, 12, 18), tuple(pmtid2omkey(205)))
        self.assertEqual((1, 11, 22), tuple(pmtid2omkey(240)))
        self.assertEqual((4, 11, 2), tuple(pmtid2omkey(1894)))
        self.assertEqual((9, 18, 0), tuple(pmtid2omkey(4465)))
        self.assertEqual((95, 7, 16), tuple(pmtid2omkey(52810)))
        self.assertEqual((95, 4, 13), tuple(pmtid2omkey(52900)))

    def test_pmtid2omkey_handles_floats(self):
        pmtid2omkey = self.det.pmtid2omkey
        self.assertEqual((1, 13, 12), tuple(pmtid2omkey(168.0)))
        self.assertEqual((1, 12, 18), tuple(pmtid2omkey(205.0)))
        self.assertEqual((1, 11, 22), tuple(pmtid2omkey(240.0)))
        self.assertEqual((4, 11, 2), tuple(pmtid2omkey(1894.0)))
        self.assertEqual((9, 18, 0), tuple(pmtid2omkey(4465.0)))
        self.assertEqual((95, 7, 16), tuple(pmtid2omkey(52810.0)))
        self.assertEqual((95, 4, 13), tuple(pmtid2omkey(52900.0)))

    def test_dom_positions(self):
        self.det.parse_doms()
        for i, position in enumerate(self.det.dom_positions):
            self.assertAlmostEqual(i + 1.1, position.x)
            self.assertAlmostEqual(i + 1.2, position.y)
            self.assertAlmostEqual(i + 1.3, position.z)


class TestPMT(TestCase):

    def test_init(self):
        pmt = PMT(1, (1, 2, 3), (4, 5, 6), 7)
        self.assertEqual(1, pmt.id)
        self.assertEqual(1, pmt.pos.x)
        self.assertEqual(2, pmt.pos.y)
        self.assertEqual(3, pmt.pos.z)
        self.assertAlmostEqual(0.455842, pmt.dir.x, 6)
        self.assertAlmostEqual(0.569803, pmt.dir.y, 6)
        self.assertAlmostEqual(0.683763, pmt.dir.z, 6)
        self.assertEqual(7, pmt.t0)
