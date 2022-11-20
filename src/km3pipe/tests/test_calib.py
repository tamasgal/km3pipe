# Filename: test_calib.py
# pylint: disable=C0111,E1003,R0904,C0103,R0201,C0102
from os.path import dirname, join
import functools
import operator
import shutil
import sys
import tempfile

import km3io
from thepipe import Module, Pipeline
import km3pipe as kp
from km3pipe.dataclasses import Table
from km3pipe.io.daq import DAQEvent
from km3pipe.hardware import Detector
from km3pipe.io.hdf5 import HDF5Sink
from km3pipe.testing import TestCase, MagicMock, patch, skip, skipif, data_path
from km3pipe.calib import Calibration, CalibrationService, slew

from .test_hardware import EXAMPLE_DETX

import numpy as np
import tables as tb

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestCalibration(TestCase):
    """Tests for the Calibration class"""

    def test_init_with_wrong_file_extension(self):
        with self.assertRaises(NotImplementedError):
            Calibration(filename="foo")

    @patch("km3pipe.calib.Detector")
    def test_init_with_filename(self, mock_detector):
        Calibration(filename="foo.detx")
        mock_detector.assert_called_with(filename="foo.detx")

    @patch("km3pipe.calib.Detector")
    def test_init_with_det_id(self, mock_detector):
        Calibration(det_id=1)
        mock_detector.assert_called_with(t0set=None, calibset=None, det_id=1)
        Calibration(det_id=1, calibset=2, t0set=3)
        mock_detector.assert_called_with(t0set=3, calibset=2, det_id=1)

    def test_init_with_detector(self):
        det = Detector(data_path("detx/detx_v1.detx"))
        Calibration(detector=det)

    def test_apply_to_hits_with_pmt_id_aka_mc_hits(self):
        calib = Calibration(filename=data_path("detx/detx_v1.detx"))

        hits = Table({"pmt_id": [1, 2, 1], "time": [10.1, 11.2, 12.3]})

        chits = calib.apply(hits, correct_slewing=False)

        assert len(hits) == len(chits)

        a_hit = chits[0]
        self.assertAlmostEqual(1.1, a_hit.pos_x)
        self.assertAlmostEqual(10, a_hit.t0)
        self.assertAlmostEqual(10.1, a_hit.time)  # t0 should not bei applied

        a_hit = chits[1]
        self.assertAlmostEqual(1.4, a_hit.pos_x)
        self.assertAlmostEqual(20, a_hit.t0)
        self.assertAlmostEqual(11.2, a_hit.time)  # t0 should not be applied

    def test_apply_to_hits_with_pmt_id_aka_mc_hits_from_km3io(self):
        calib = Calibration(filename=data_path("detx/KM3NeT_-00000001_20171212.detx"))
        f = km3io.OfflineReader(
            data_path(
                "offline/mcv6.gsg_nue-CCHEDIS_1e4-1e6GeV.sirene.jte.jchain.aanet.1.root"
            )
        )

        for event in f:
            chits = calib.apply(event.mc_hits)
            assert 840 == len(chits.t0)
            assert np.allclose([3, 26, 24, 4, 23, 25], chits.channel_id[:6])
            assert np.allclose([3401, 3401, 3406, 3411, 5501, 5501], chits.dom_id[:6])
            assert np.allclose([1, 1, 6, 11, 1, 1], chits.floor[:6])
            assert np.allclose([34, 34, 34, 34, 55, 55], chits.du[:6])
            assert np.allclose(
                [
                    1679.18706571,
                    1827.14262054,
                    1926.71722628,
                    2433.83097585,
                    1408.35942832,
                    1296.51397496,
                ],
                chits.time[:6],
            )
            assert np.allclose(
                [2.034, 1.847, 1.938, 2.082, -54.96, -55.034], chits.pos_x[:6]
            )
            assert np.allclose(
                [-233.415, -233.303, -233.355, -233.333, -341.346, -341.303],
                chits.pos_y[:6],
            )
            assert np.allclose(
                [65.059, 64.83, 244.83, 425.111, 64.941, 64.83], chits.pos_z[:6]
            )
            assert np.allclose([4, 4, 4, 26, 4, 4], f.mc_hits.origin[0][:6].tolist())
            assert np.allclose(
                [36835, 36881, 37187, 37457, 60311, 60315],
                f.mc_hits.pmt_id[0][:6].tolist(),
            )
            break

    def test_dus(self):
        calib = Calibration(filename=data_path("detx/detx_v1.detx"))

        hits = Table(
            {"dom_id": [2, 6, 3], "channel_id": [0, 1, 2], "time": [10.1, 11.2, 12.3]}
        )

        dus = calib.dus(hits)
        assert np.allclose([1, 2, 1], dus)

    def test_floors(self):
        calib = Calibration(filename=data_path("detx/detx_v1.detx"))

        hits = Table(
            {"dom_id": [2, 6, 3], "channel_id": [0, 1, 2], "time": [10.1, 11.2, 12.3]}
        )

        floors = calib.floors(hits)
        assert np.allclose([2, 3, 3], floors)

    def test_apply_to_hits_with_dom_id_and_channel_id(self):
        calib = Calibration(filename=data_path("detx/detx_v1.detx"))

        hits = Table(
            {"dom_id": [2, 3, 3], "channel_id": [0, 1, 2], "time": [10.1, 11.2, 12.3]}
        )

        chits = calib.apply(hits, correct_slewing=False)

        assert len(hits) == len(chits)

        a_hit = chits[0]
        self.assertAlmostEqual(2.1, a_hit.pos_x)
        self.assertAlmostEqual(40, a_hit.t0)
        t0 = a_hit.t0
        self.assertAlmostEqual(10.1 + t0, a_hit.time)

        a_hit = chits[1]
        self.assertAlmostEqual(3.4, a_hit.pos_x)
        self.assertAlmostEqual(80, a_hit.t0)
        t0 = a_hit.t0
        self.assertAlmostEqual(11.2 + t0, a_hit.time)

    def test_assert_apply_adds_dom_id_and_channel_id_to_mc_hits(self):
        calib = Calibration(filename=data_path("detx/detx_v1.detx"))
        hits = Table({"pmt_id": [1, 2, 1], "time": [10.1, 11.2, 12.3]})
        chits = calib.apply(hits)
        self.assertListEqual([1, 1, 1], list(chits.dom_id))
        self.assertListEqual([0, 1, 0], list(chits.channel_id))

    def test_assert_apply_adds_pmt_id_to_hits(self):
        calib = Calibration(filename=data_path("detx/detx_v1.detx"))
        hits = Table(
            {"dom_id": [2, 3, 3], "channel_id": [0, 1, 2], "time": [10.1, 11.2, 12.3]}
        )
        chits = calib.apply(hits, correct_slewing=False)
        self.assertListEqual([4, 8, 9], list(chits.pmt_id))

    def test_apply_to_hits_with_pmt_id_with_wrong_calib_raises(self):
        calib = Calibration(filename=data_path("detx/detx_v1.detx"))

        hits = Table({"pmt_id": [999], "time": [10.1]})

        with self.assertRaises(KeyError):
            calib.apply(hits, correct_slewing=False)

    def test_apply_to_hits_with_dom_id_and_channel_id_with_wrong_calib_raises(self):
        calib = Calibration(filename=data_path("detx/detx_v1.detx"))

        hits = Table({"dom_id": [999], "channel_id": [0], "time": [10.1]})

        with self.assertRaises(KeyError):
            calib.apply(hits, correct_slewing=False)

    def test_apply_to_hits_from_km3io(self):
        calib = Calibration(filename=data_path("detx/km3net_offline.detx"))
        hits = km3io.OfflineReader(data_path("offline/km3net_offline.root"))[0].hits

        chits = calib.apply(hits)
        assert 176 == len(chits.t0)
        assert np.allclose([207747.825, 207745.656, 207743.836], chits.t0.tolist()[:3])

        chits = calib.apply(hits[:3])
        assert 3 == len(chits.t0)
        assert np.allclose([207747.825, 207745.656, 207743.836], chits.t0.tolist()[:3])

    def test_apply_to_hits_from_km3io_iterator(self):
        calib = Calibration(filename=data_path("detx/km3net_offline.detx"))
        f = km3io.OfflineReader(data_path("offline/km3net_offline.root"))

        for event in f:
            chits = calib.apply(event.hits)
            assert 176 == len(chits.t0)
            assert np.allclose(
                [207747.825, 207745.656, 207743.836], chits.t0.tolist()[:3]
            )
            break

    def test_daq_triggered_hits(self):
        calib = Calibration(filename=data_path("detx/detx_v1.detx"))

        dt = DAQEvent.triggered_hits_dt_final

        raw_hits = np.array(
            [(2, 0, 10, 0, 100), (3, 1, 11, 10, 200), (3, 2, 12, 255, 300)],
            dtype=dt,
        )
        print(raw_hits.dtype)

        hits = Table(raw_hits)

        chits = calib.apply(hits)  #  correct_slewing=True is default

        assert len(hits) == len(chits)

        a_hit = chits[0]
        self.assertAlmostEqual(10 + a_hit.t0 - slew(a_hit.tot), a_hit.time, places=5)

        a_hit = chits[1]
        self.assertAlmostEqual(11 + a_hit.t0 - slew(a_hit.tot), a_hit.time, places=5)

        a_hit = chits[2]
        self.assertAlmostEqual(12 + a_hit.t0 - slew(a_hit.tot), a_hit.time, places=5)

    def test_time_slewing_correction(self):
        calib = Calibration(filename=data_path("detx/detx_v1.detx"))

        hits = Table(
            {
                "dom_id": [2, 3, 3],
                "channel_id": [0, 1, 2],
                "time": [10.1, 11.2, 12.3],
                "tot": [0, 10, 255],
            }
        )

        chits = calib.apply(hits)  #  correct_slewing=True is default

        assert len(hits) == len(chits)

        a_hit = chits[0]
        self.assertAlmostEqual(10.1 + a_hit.t0 - slew(a_hit.tot), a_hit.time)

        a_hit = chits[1]
        self.assertAlmostEqual(11.2 + a_hit.t0 - slew(a_hit.tot), a_hit.time)

        a_hit = chits[2]
        self.assertAlmostEqual(12.3 + a_hit.t0 - slew(a_hit.tot), a_hit.time)

    def test_apply_to_timeslice_hits(self):
        tshits = Table.from_template(
            {
                "channel_id": [0, 1, 2],
                "dom_id": [2, 3, 3],
                "time": [10.1, 11.2, 12.3],
                "tot": np.ones(3, dtype=float),
                "group_id": 0,
            },
            "TimesliceHits",
        )
        calib = Calibration(filename=data_path("detx/detx_v1.detx"))
        c_tshits = calib.apply(tshits, correct_slewing=False)
        assert len(c_tshits) == len(tshits)
        assert np.allclose([40, 80, 90], c_tshits.t0)
        # TimesliceHits is using int4 for times, so it's truncated when we pass in float64
        assert np.allclose([50.1, 91.2, 102.3], c_tshits.time, atol=0.1)

    def test_apply_without_affecting_primary_hit_table(self):
        calib = Calibration(filename=data_path("detx/detx_v1.detx"))
        hits = Table({"pmt_id": [1, 2, 1], "time": [10.1, 11.2, 12.3]})
        hits_compare = hits.copy()
        calib.apply(hits, correct_slewing=False)

        for t_primary, t_calib in zip(hits_compare, hits):
            self.assertAlmostEqual(t_primary, t_calib)

    def test_calibration_in_pipeline(self):
        class DummyPump(kp.Module):
            def configure(self):
                self.index = 0

            def process(self, blob):
                self.index += 1
                mc_hits = Table({"pmt_id": [1, 2, 1], "time": [10.1, 11.2, 12.3]})
                hits = Table(
                    {
                        "dom_id": [2, 3, 3],
                        "channel_id": [0, 1, 2],
                        "time": [10.1, 11.2, 12.3],
                        "tot": [0, 10, 255],
                    }
                )

                blob["Hits"] = hits
                blob["McHits"] = mc_hits
                return blob

        _self = self

        class Observer(kp.Module):
            def process(self, blob):
                assert "Hits" in blob
                assert "McHits" in blob
                assert "CalibHits" in blob
                assert "CalibMcHits" in blob
                assert not hasattr(blob["Hits"], "pmt_id")
                assert hasattr(blob["CalibHits"], "pmt_id")
                assert not hasattr(blob["McHits"], "dom_id")
                assert hasattr(blob["CalibHits"], "dom_id")
                assert np.allclose([10.1, 11.2, 12.3], blob["Hits"].time)
                assert np.allclose([42.09, 87.31, 111.34], blob["CalibHits"].time)
                assert np.allclose(blob["McHits"].time, blob["CalibMcHits"].time)
                return blob

        pipe = kp.Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(Calibration, filename=data_path("detx/detx_v1.detx"))
        pipe.attach(Observer)
        pipe.drain(3)


class TestCalibrationService(TestCase):
    def test_apply_to_hits_with_dom_id_and_channel_id(self):

        hits = Table(
            {
                "dom_id": [2, 3, 3],
                "channel_id": [0, 1, 2],
                "time": [10.1, 11.2, 12.3],
                "tot": [23, 105, 231],
            }
        )

        tester = self

        class HitCalibrator(Module):
            def process(self, blob):
                chits = self.services["calibrate"](hits)

                assert len(hits) == len(chits)

                a_hit = chits[0]
                tester.assertAlmostEqual(2.1, a_hit.pos_x)
                tester.assertAlmostEqual(40, a_hit.t0)
                t0 = a_hit.t0
                tester.assertAlmostEqual(10.1 + t0 - slew(a_hit.tot), a_hit.time)

                a_hit = chits[1]
                tester.assertAlmostEqual(3.4, a_hit.pos_x)
                tester.assertAlmostEqual(80, a_hit.t0)
                t0 = a_hit.t0
                tester.assertAlmostEqual(11.2 + t0 - slew(a_hit.tot), a_hit.time)
                return blob

        pipe = Pipeline()
        pipe.attach(CalibrationService, filename=data_path("detx/detx_v1.detx"))
        pipe.attach(HitCalibrator)
        pipe.drain(1)

    def test_apply_to_hits_with_dom_id_and_channel_id_without_slewing(self):

        hits = Table(
            {
                "dom_id": [2, 3, 3],
                "channel_id": [0, 1, 2],
                "time": [10.1, 11.2, 12.3],
                "tot": [23, 105, 231],
            }
        )

        tester = self

        class HitCalibrator(Module):
            def process(self, blob):
                chits = self.services["calibrate"](hits, correct_slewing=False)

                assert len(hits) == len(chits)

                a_hit = chits[0]
                tester.assertAlmostEqual(2.1, a_hit.pos_x)
                tester.assertAlmostEqual(40, a_hit.t0)
                t0 = a_hit.t0
                tester.assertAlmostEqual(10.1 + t0, a_hit.time)

                a_hit = chits[1]
                tester.assertAlmostEqual(3.4, a_hit.pos_x)
                tester.assertAlmostEqual(80, a_hit.t0)
                t0 = a_hit.t0
                tester.assertAlmostEqual(11.2 + t0, a_hit.time)
                return blob

        pipe = Pipeline()
        pipe.attach(CalibrationService, filename=data_path("detx/detx_v1.detx"))
        pipe.attach(HitCalibrator)
        pipe.drain(1)

    def test_correct_slewing(self):

        hits = Table(
            {
                "dom_id": [2, 3, 3],
                "channel_id": [0, 1, 2],
                "time": [10.1, 11.2, 12.3],
                "tot": [0, 10, 255],
            }
        )

        tester = self

        class HitCalibrator(Module):
            def process(self, blob):
                self.services["correct_slewing"](hits)

                a_hit = hits[0]
                tester.assertAlmostEqual(10.1 - slew(a_hit.tot), a_hit.time)

                a_hit = hits[1]
                tester.assertAlmostEqual(11.2 - slew(a_hit.tot), a_hit.time)
                return blob

        pipe = Pipeline()
        pipe.attach(CalibrationService, filename=data_path("detx/detx_v1.detx"))
        pipe.attach(HitCalibrator)
        pipe.drain(1)

    def test_provided_detector_data(self):
        class DetectorReader(Module):
            def process(self, blob):
                assert "get_detector" in self.services
                det = self.services["get_detector"]()
                assert isinstance(det, Detector)

        pipe = Pipeline()
        pipe.attach(CalibrationService, filename=data_path("detx/detx_v1.detx"))
        pipe.attach(DetectorReader)
        pipe.drain(1)


class TestSlew(TestCase):
    def test_slew(self):
        self.assertAlmostEqual(8.01, slew(0))
        self.assertAlmostEqual(0.60, slew(23))
        self.assertAlmostEqual(-9.04, slew(255))

    def test_slew_vectorised(self):
        assert np.allclose([8.01, 0.60, -9.04], slew(np.array([0, 23, 255])))
