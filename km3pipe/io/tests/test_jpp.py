# Filename: test_jpp.py
# pylint: disable=locally-disabled,C0111,R0904,C0301,C0103,W0212
from km3pipe.testing import TestCase, surrogate, patch
import sys
import numpy as np
from km3pipe.io.jpp import EventPump, TimeslicePump, SummaryslicePump
from km3pipe.io.hdf5 import HDF5Pump
from os.path import join, dirname, abspath

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2018, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

TESTDATA_DIR = abspath(join(dirname(__file__), "../../kp-data/test_data"))


class TestEventPump(TestCase):
    def setUp(self):
        jpp_fname = join(TESTDATA_DIR, 'a.root')
        h5_fname = join(TESTDATA_DIR, 'a.h5')
        self.jpp_pump = EventPump(filename=jpp_fname)
        self.h5_pump = HDF5Pump(filename=h5_fname)

    def test_event_info(self):
        n = self.jpp_pump.n_events
        for i in range(n):
            h5blob = self.h5_pump[i]
            jppblob = self.jpp_pump[i]
            for k in h5blob["EventInfo"].dtype.names:
                if k not in jppblob["EventInfo"].dtype.names:
                    continue
                ref_value = h5blob["EventInfo"][k][0]
                test_value = jppblob["EventInfo"][k][0]
                if isinstance(ref_value, float):
                    if np.isnan(ref_value):
                        assert np.isnan(test_value) == np.isnan(ref_value)
                    else:
                        self.assertAlmostEqual(test_value, ref_value)
                else:
                    assert ref_value == test_value

    def test_hit_info(self):
        n = self.jpp_pump.n_events
        for i in range(n):
            h5blob = self.h5_pump[i]
            jppblob = self.jpp_pump[i]
            for k in h5blob["Hits"].dtype.names:
                ref_value = h5blob["Hits"][k][0]
                test_value = jppblob["Hits"][k][0]
                if isinstance(ref_value, float):
                    if np.isnan(ref_value):
                        assert np.isnan(test_value)
                    else:
                        self.assertAlmostEqual(test_value, ref_value)
                else:
                    assert ref_value == test_value


class TestTimeslicePump(TestCase):
    def setUp(self):
        self.jpp_fname = join(TESTDATA_DIR, 'a.root')

    def test_header(self):
        timeslice_pump = TimeslicePump(filename=self.jpp_fname, stream="L1")
        blob = timeslice_pump[0]
        tsinfo = blob["TimesliceInfo"]
        assert tsinfo.frame_index == 366
        assert tsinfo.slice_id == 0
        assert tsinfo.timestamp == 1575979236
        assert tsinfo.nanoseconds == 600000000
        assert tsinfo.n_frames == 69


# class TestSummaryslicePump(TestCase):
#     @surrogate('jppy.daqsummaryslicereader.PyJDAQSummarysliceReader')
#     @patch('jppy.daqsummaryslicereader.PyJDAQSummarysliceReader')
#     def test_init(self, reader_mock):
#         filename = 'a.root'
#         SummaryslicePump(filename=filename)
#         reader_mock.assert_called_with(filename.encode())
