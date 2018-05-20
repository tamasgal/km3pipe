# Filename: no_test_daq.py
# pylint: disable=C0111,R0904,C0103
"""
Tests for KM3NeT binary formats readout.

"""
from os.path import dirname, join

from km3pipe.testing import TestCase
from km3pipe.io.daq import (DAQPump, DAQPreamble, DAQHeader, DAQSummaryslice)

TEST_DATA_DIR = join(dirname(__file__), "../../kp-data/test_data")
IO_SUM_FILE = join(TEST_DATA_DIR, "IO_SUM.dat")
IO_EVT_FILE = join(TEST_DATA_DIR, "IO_EVT.dat")


class TestDAQPump(TestCase):
    def test_init(self):
        DAQPump()

    def test_init_with_filename(self):
        DAQPump(IO_SUM_FILE)

    def test_frame_positions_in_io_sum(self):
        p = DAQPump(IO_SUM_FILE)
        assert 81 == len(p.frame_positions)
        self.assertListEqual([0, 656, 1312], p.frame_positions[:3])
        self.assertListEqual([50973, 51629, 52285], p.frame_positions[-3:])

    def test_frame_positions_in_io_evt(self):
        p = DAQPump(IO_EVT_FILE)
        assert 38 == len(p.frame_positions)
        self.assertListEqual([0, 570, 986], p.frame_positions[:3])
        self.assertListEqual([13694, 14016, 14360], p.frame_positions[-3:])

    def test_blob_in_io_sum(self):
        p = DAQPump(IO_SUM_FILE)
        blob = p.next_blob()
        assert 'DAQSummaryslice' in blob.keys()
        assert 'DAQPreamble' in blob.keys()
        assert 'DAQHeader' in blob.keys()
        assert 16 == blob['DAQSummaryslice'].n_summary_frames

    def test_blob_in_io_evt(self):
        p = DAQPump(IO_EVT_FILE)
        blob = p.next_blob()
        assert 'DAQEvent' in blob.keys()
        assert 'DAQPreamble' in blob.keys()
        assert 'DAQHeader' in blob.keys()
        event = blob['DAQEvent']
        assert 13 == event.n_triggered_hits
        assert 28 == event.n_snapshot_hits

    def test_blob_iteration(self):
        p = DAQPump(IO_EVT_FILE)
        for blob in p:
            pass

    def test_get_item(self):
        p = DAQPump(IO_EVT_FILE)
        blob = p[4]
        event = blob['DAQEvent']
        assert 6 == event.n_triggered_hits
        assert 17 == event.n_snapshot_hits
