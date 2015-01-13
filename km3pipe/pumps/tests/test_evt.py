from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

from km3pipe.testing import *
from km3pipe.pumps import EvtPump


class TestParser(TestCase):

    def setUp(self):
        self.valid_evt_header = "\n".join((
            "start_run: 1",
            "cut_nu: 0.100E+03 0.100E+09-0.100E+01 0.100E+01",
            "spectrum: -1.40",
            "end_event:",
            "start_event:",
            "track_in: 1 -389.951 213.427 516 -0.204562 -0.60399 -0.770293 9.092 0 5 40.998",
            "hit: 1 44675 1 1170.59 5 2 1 1170.59",
            "end_event:"
        ))
        self.corrupt_evt_header = "foo"

    def test_parse_header(self):
        self.temp_file = StringIO(self.valid_evt_header)
        self.pump = EvtPump()
        self.pump.blob_file = self.temp_file
        raw_header = self.pump.extract_header()
        self.assertEqual(['1'], raw_header['start_run'])
        self.assertAlmostEqual(-1.4, float(raw_header['spectrum'][0]))
        self.assertAlmostEqual(1, float(raw_header['cut_nu'][2]))
        self.temp_file.close()

    def test_incomplete_header_raises_value_error(self):
        self.temp_file = StringIO(self.corrupt_evt_header)
        self.pump = EvtPump()
        self.pump.blob_file = self.temp_file
        with self.assertRaises(ValueError):
            self.pump.extract_header()
        self.temp_file.close()

    def test_event_offset_is_at_first_event_after_parsing_header(self):
        self.temp_file = StringIO(self.valid_evt_header)
        self.pump = EvtPump()
        self.pump.blob_file = self.temp_file
        raw_header = self.pump.extract_header()
        self.assertEqual(88, self.pump.event_offsets[0])
        self.temp_file.close()
