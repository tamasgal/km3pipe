# Filename: no_test_daq.py
# pylint: disable=C0111,R0904,C0103
"""
Tests for KM3NeT binary formats readout.

"""
from os.path import dirname, join

import numpy as np

from km3pipe.testing import TestCase
from km3pipe.io.daq import (
    DAQPump, DAQPreamble, DAQHeader, DAQSummaryslice, DMMonitor, TMCHRepump
)

TEST_DATA_DIR = join(dirname(__file__), "../../kp-data/test_data")
IO_SUM_FILE = join(TEST_DATA_DIR, "IO_SUM.dat")
IO_EVT_FILE = join(TEST_DATA_DIR, "IO_EVT.dat")
IO_MONIT_FILE = join(TEST_DATA_DIR, "IO_MONIT.dat")


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


class TestTMCHRepump(TestCase):
    def test_reading_version_2(self):
        repump = TMCHRepump(filename=IO_MONIT_FILE)
        packets = [p['TMCHData'] for p in repump]

        p1 = packets[0]
        p2 = packets[5]

        assert 86 == p1.run
        assert 0 == p1.udp_sequence_number
        assert 541 == p1.utc_seconds
        assert 500000000 == p1.nanoseconds
        assert 806472270 == p1.dom_id
        assert 2 == p1.version
        self.assertAlmostEqual(199.05982971191406, p1.yaw)
        self.assertAlmostEqual(0.5397617816925049, p1.pitch)
        self.assertAlmostEqual(-0.2243121862411499, p1.roll)
        self.assertAlmostEqual(32.35, p1.temp)
        self.assertAlmostEqual(16.77, p1.humidity)
        assert np.allclose(np.full(31, 0), p1.pmt_rates)
        assert np.allclose([0.00708725, 0.00213623, -0.86456668], p1.A)
        assert np.allclose([-0.2621212, 0.02363636, 0.1430303], p1.H)
        assert np.allclose([-2.87721825, -0.83284622, -0.28969574], p1.G)

        assert 86 == p2.run
        assert 0 == p2.udp_sequence_number
        assert 542 == p2.utc_seconds
        assert 0 == p2.nanoseconds
        assert 806472270 == p2.dom_id
        assert 2 == p2.version
        self.assertAlmostEqual(199.05982971191406, p2.yaw)
        self.assertAlmostEqual(0.5397617816925049, p2.pitch)
        self.assertAlmostEqual(-0.2243121862411499, p2.roll)
        self.assertAlmostEqual(32.35, p2.temp)
        self.assertAlmostEqual(16.77, p2.humidity)
        assert np.allclose(np.full(31, 0), p2.pmt_rates)
        assert np.allclose([0.00708725, 0.00213623, -0.86456668], p2.A)
        assert np.allclose([-0.2621212, 0.02363636, 0.1430303], p2.H)
        assert np.allclose([-2.87721825, -0.83284622, -0.28969574], p2.G)


class TestDMMonitor(TestCase):
    def test_init(self):
        dmm = DMMonitor('a')
        assert 'http://a:1302/mon/' == dmm._url

    def test_available_parameters(self):
        dmm = DMMonitor('a')
        dmm._available_parameters = ['b', 'c']
        self.assertListEqual(['b', 'c'], dmm.available_parameters)
