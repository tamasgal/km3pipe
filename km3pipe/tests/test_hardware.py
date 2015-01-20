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
from km3pipe.hardware import Detector

example_detx = StringIO("\n".join((
    "1 3",
    "1 1 1 3",
    " 1 1.1 1.2 1.3 -1.1  0.2  0.3 10",
    " 2 1.1 1.2 1.3  0.1 -1.2  0.3 20",
    " 3 1.1 1.2 1.3  0.1  0.2 -1.3 30",
    "2 1 2 3",
    " 4 1.1 1.2 2.3 -1.1  0.2  0.3 40",
    " 5 1.1 1.2 2.3  0.1 -1.2  0.3 50",
    " 6 1.1 1.2 2.3  0.1  0.2 -1.3 60",
    "3 1 3 3",
    " 7 1.1 1.2 3.3 -1.1  0.2  0.3 70",
    " 8 1.1 1.2 3.3  0.1 -1.2  0.3 80",
    " 9 1.1 1.2 3.3  0.1  0.2 -1.3 90",)))

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
        self.assertTupleEqual((7, 1.1, 1.2, 3.3, -1.1,  0.2,  0.3, 70),
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
