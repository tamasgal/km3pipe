__author__ = 'tamasgal'

import unittest

try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

from km3pipe.pumps import EvtPump


class TestParser(unittest.TestCase):

    def setUp(self):
        self.valid_evt_header = "\n".join((
            "start_run: 1",
            "cut_nu: 0.100E+03 0.100E+09-0.100E+01 0.100E+01",
            "spectrum: -1.40",
            "end_event:"
        ))
        self.corrupt_evt_header = "foo"

    def test_parse_header(self):
        self.temp_file = StringIO(self.valid_evt_header)
        self.pump = EvtPump()
        self.pump.evt_file = self.temp_file
        raw_header = self.pump.extract_header()
        self.assertEqual(['1'], raw_header['start_run'])
        self.assertAlmostEqual(-1.4, float(raw_header['spectrum'][0]))
        self.assertAlmostEqual(1, float(raw_header['cut_nu'][2]))
        self.temp_file.close()

    def test_incomplete_header_raises_value_error(self):
        self.temp_file = StringIO(self.corrupt_evt_header)
        self.pump = EvtPump()
        self.pump.evt_file = self.temp_file
        with self.assertRaises(ValueError):
            self.pump.extract_header()
        self.temp_file.close()
