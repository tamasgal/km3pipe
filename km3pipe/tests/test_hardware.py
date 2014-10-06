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
    " 1 1 1 1 -1  0  0 10",
    " 2 1 1 1  0 -1  0 20",
    " 3 1 1 1  0  0 -1 30",
    "2 1 2 3",
    " 4 1 1 2 -1  0  0 40",
    " 5 1 1 2  0 -1  0 50",
    " 6 1 1 2  0  0 -1 60",
    "3 1 3 3",
    " 7 1 1 3 -1  0  0 70",
    " 8 1 1 3  0 -1  0 80",
    " 9 1 1 3  0  0 -1 90",)))

class TestHardware(TestCase):

    def test_detector_id(self):
        det = Detector()
        det.det_file = example_detx
        self.assertEqual(1, det.id)
