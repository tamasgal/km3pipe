# Filename: test_online.py
# pylint: disable=locally-disabled,C0111,R0904,C0301,C0103,W0212
from km3pipe.testing import TestCase, patch, skip, data_path
import sys
import numpy as np
from km3pipe.io.online import EventPump, TimeslicePump, SummaryslicePump
from km3pipe.io.hdf5 import HDF5Pump
from os.path import join, dirname, abspath

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2018, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal, Johannes Schumann"
__email__ = "tgal@km3net.de"
__status__ = "Development"

ONLINE_TEST_FILE = data_path("online/events_sample.root")
ONLINE_TEST_FILE_H5 = data_path("online/events_sample.h5")


class TestEventPump(TestCase):
    def setUp(self):
        self.jpp_pump = EventPump(filename=ONLINE_TEST_FILE)
        self.h5_pump = HDF5Pump(filename=ONLINE_TEST_FILE_H5)

    def tearDown(self):
        self.jpp_pump.finish()
        self.h5_pump.finish()

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

    @skip
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
        self.jpp_fname = ONLINE_TEST_FILE
        self.timeslice_pump = TimeslicePump(filename=self.jpp_fname, stream="L1")

    def test_timeslice_info(self):
        blob = self.timeslice_pump[0]
        tsinfo = blob["TimesliceInfo"]
        assert tsinfo.frame_index == 366
        assert tsinfo.slice_id == 0
        assert tsinfo.timestamp == 1575979236
        assert tsinfo.nanoseconds == 600000000
        assert tsinfo.n_frames == 69

    def test_timeslice_hits(self):
        blob = self.timeslice_pump[0]
        l1hits = blob["L1Hits"]
        counts = [
            1421,
            1272,
            1533,
            1209,
            1311,
            1450,
            1102,
            1494,
            1410,
            1390,
            1225,
            1419,
            1474,
            1452,
            1589,
            1362,
            1577,
            1598,
            1891,
            1584,
            1571,
            1523,
            1696,
            1719,
            1540,
            1578,
            1656,
            1569,
            1422,
            1401,
            1262,
        ]
        tot = [
            216,
            681,
            1222,
            1580,
            1484,
            1221,
            940,
            887,
            908,
            950,
            844,
            736,
            717,
            661,
            752,
            716,
            791,
            946,
            1089,
            1220,
            1484,
            1786,
            2265,
            2539,
            2932,
            3132,
            3201,
            2901,
            2403,
            1654,
            1056,
            566,
            297,
            167,
            94,
            74,
            82,
            60,
            50,
            37,
            30,
            24,
            24,
            20,
            19,
            12,
            15,
            15,
            13,
            8,
            5,
            6,
            9,
            9,
            7,
            9,
            11,
            6,
            5,
            10,
            7,
            4,
            7,
            2,
            7,
            1,
            4,
            5,
            5,
            3,
            3,
            3,
            2,
            3,
            2,
            3,
            1,
            4,
            3,
            4,
            1,
            1,
            2,
            2,
            1,
            2,
            3,
            1,
            1,
            1,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
        dom_id = [
            774,
            244,
            602,
            935,
            533,
            878,
            862,
            699,
            760,
            1058,
            625,
            641,
            621,
            679,
            912,
            461,
            925,
            633,
            754,
            701,
            767,
            784,
            812,
            299,
            605,
            1010,
            226,
            892,
            785,
            864,
            707,
            540,
            749,
            772,
            172,
            748,
            773,
            585,
            278,
            656,
            852,
            685,
            1041,
            473,
            639,
            695,
            837,
            830,
            656,
            676,
            486,
            647,
            533,
            716,
            649,
            599,
            502,
            271,
            629,
            769,
            613,
            784,
            705,
            600,
            984,
            682,
            243,
            583,
        ]
        self.assertListEqual(
            counts, np.unique(l1hits.channel_id, return_counts=True)[1].tolist()
        )
        self.assertListEqual(tot, np.unique(l1hits.tot, return_counts=True)[1].tolist())
        self.assertListEqual(
            dom_id, np.unique(l1hits.dom_id, return_counts=True)[1].tolist()
        )


class TestSummaryslicePump(TestCase):
    def setUp(self):
        self.pump = SummaryslicePump(filename=ONLINE_TEST_FILE)

    def test_summaryslice_info(self):
        sum_slice_info = self.pump[0]["SummarysliceInfo"]
        assert sum_slice_info.frame_index == 183
        assert sum_slice_info.timestamp == 1575979218
        assert sum_slice_info.nanoseconds == 300000000 / 16
        assert sum_slice_info.n_frames == 2

    def test_summaryslice_field_len(self):
        sum_slice = self.pump[0]["Summaryslice"]
        dom_ids = [808447180, 808488895]
        self.assertListEqual(dom_ids, list(sum_slice.keys()))
        for i in range(2):
            assert len(sum_slice[dom_ids[i]]["rates"]) == 31
            assert len(sum_slice[dom_ids[i]]["fifos"]) == 31
            assert len(sum_slice[dom_ids[i]]["hrvs"]) == 31

    def test_summaryslice_ctnt(self):
        sum_slice = self.pump[0]["Summaryslice"]
        dom_ids = [808447180, 808488895]
        assert sum_slice[dom_ids[0]]["max_sequence_number"] == 26
        assert sum_slice[dom_ids[0]]["n_udp_packets"] == 27
        assert sum_slice[dom_ids[0]]["has_udp_trailer"]
        assert not sum_slice[dom_ids[0]]["fifo_status"]
        self.assertAlmostEqual(
            sum_slice[dom_ids[0]]["rates"][6],
            6586.0,
        )
        self.assertAlmostEqual(sum_slice[dom_ids[1]]["rates"][4], 5752.0)
