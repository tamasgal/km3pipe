# Filename: test_calib.py
# pylint: disable=C0111,E1003,R0904,C0103,R0201,C0102
from os.path import dirname, join
import functools
import operator
import shutil
import sys
import tempfile

from km3pipe.core import Module, Pipeline
from km3pipe.dataclasses import Table
from km3pipe.hardware import Detector
from km3pipe.io.hdf5 import HDF5Sink
from km3pipe.testing import TestCase, MagicMock, patch, skip, skipif
from km3pipe.calib import Calibration, CalibrationService
from km3pipe.utils import calibrate

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

DATA_DIR = join(dirname(__file__), '../kp-data/test_data/')
DETX_FILENAME = join(DATA_DIR, 'detx_v1.detx')


class TestCalibration(TestCase):
    """Tests for the Calibration class"""

    def test_init_with_wrong_file_extension(self):
        with self.assertRaises(NotImplementedError):
            Calibration(filename='foo')

    @patch('km3pipe.calib.Detector')
    def test_init_with_filename(self, mock_detector):
        Calibration(filename='foo.detx')
        mock_detector.assert_called_with(filename='foo.detx')

    @patch('km3pipe.calib.Detector')
    def test_init_with_det_id(self, mock_detector):
        Calibration(det_id=1)
        mock_detector.assert_called_with(
            t0set=None, calibration=None, det_id=1
        )
        Calibration(det_id=1, calibration=2, t0set=3)
        mock_detector.assert_called_with(t0set=3, calibration=2, det_id=1)

    def test_init_with_detector(self):
        det = Detector(DETX_FILENAME)
        Calibration(detector=det)

    def test_apply_to_hits_with_pmt_id(self):
        calib = Calibration(filename=DETX_FILENAME)

        hits = Table({'pmt_id': [1, 2, 1], 'time': [10.1, 11.2, 12.3]})

        chits = calib.apply(hits)

        assert len(hits) == len(chits)

        a_hit = chits[0]
        self.assertAlmostEqual(1.1, a_hit.pos_x)
        self.assertAlmostEqual(10, a_hit.t0)
        t0 = a_hit.t0
        self.assertAlmostEqual(10.1 + t0, a_hit.time)

        a_hit = chits[1]
        self.assertAlmostEqual(1.4, a_hit.pos_x)
        self.assertAlmostEqual(20, a_hit.t0)
        t0 = a_hit.t0
        self.assertAlmostEqual(11.2 + t0, a_hit.time)

    def test_apply_to_hits_with_dom_id_and_channel_id(self):
        calib = Calibration(filename=DETX_FILENAME)

        hits = Table({
            'dom_id': [2, 3, 3],
            'channel_id': [0, 1, 2],
            'time': [10.1, 11.2, 12.3]
        })

        chits = calib.apply(hits)

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

    def test_apply_to_timeslice_hits(self):
        tshits = Table.from_template({
            'channel_id': [0, 1, 2],
            'dom_id': [2, 3, 3],
            'time': [10.1, 11.2, 12.3],
            'tot': np.ones(3, dtype=float),
            'group_id': 0,
        }, 'TimesliceHits')
        calib = Calibration(filename=DETX_FILENAME)
        c_tshits = calib.apply(tshits)
        assert len(c_tshits) == len(tshits)
        assert np.allclose([40, 80, 90], c_tshits.t0)
        # TimesliceHits is using int4 for times, so it's truncated when we pass in float64
        assert np.allclose([50.1, 91.2, 102.3], c_tshits.time, atol=0.1)

    def test_apply_without_affecting_primary_hit_table(self):
        calib = Calibration(filename=DETX_FILENAME)
        hits = Table({'pmt_id': [1, 2, 1], 'time': [10.1, 11.2, 12.3]})
        hits_compare = hits.copy()
        calib.apply(hits)

        for t_primary, t_calib in zip(hits_compare, hits):
            self.assertAlmostEqual(t_primary, t_calib)


class TestCalibrationService(TestCase):
    def test_apply_to_hits_with_dom_id_and_channel_id(self):

        hits = Table({
            'dom_id': [2, 3, 3],
            'channel_id': [0, 1, 2],
            'time': [10.1, 11.2, 12.3]
        })

        tester = self

        class HitCalibrator(Module):
            def process(self, blob):
                chits = self.services['calibrate'](hits)

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
        pipe.attach(CalibrationService, filename=DETX_FILENAME)
        pipe.attach(HitCalibrator)
        pipe.drain(1)

    def test_provided_detector_data(self):
        class DetectorReader(Module):
            def process(self, blob):
                assert 'get_detector' in self.services
                det = self.services['get_detector']()
                assert isinstance(det, Detector)

        pipe = Pipeline()
        pipe.attach(CalibrationService, filename=DETX_FILENAME)
        pipe.attach(DetectorReader)
        pipe.drain(1)


@skipif(
    sys.version_info < (3, 2),
    reason="TemporaryDirectory context manager not available in Python <3.2"
)
class TestCalibrationUtility(TestCase):
    def test_consistency(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            outfile = join(tmpdirname, "out.h5")

            class DummyPump(Module):
                def configure(self):
                    self.n_total_hits = 0
                    self.original_times = []

                def process(self, blob):
                    n_hits = np.random.randint(10, 20)
                    times = np.random.rand(n_hits) * 1000
                    blob['Hits'] = Table({
                        "dom_id": np.random.randint(1, 7, n_hits),
                        "channel_id": np.random.randint(0, 3, n_hits),
                        "time": times
                    },
                                         h5loc="/hits",
                                         split_h5=True)
                    self.n_total_hits += n_hits
                    self.original_times.append(times)
                    return blob

                def finish(self):
                    return {
                        "n_total_hits": self.n_total_hits,
                        "original_times": self.original_times
                    }

            pipe = Pipeline()
            pipe.attach(DummyPump)
            pipe.attach(HDF5Sink, filename=outfile)
            result = pipe.drain(10)

            n_total_hits = result['DummyPump']["n_total_hits"]
            original_times = functools.reduce(
                operator.iconcat, result['DummyPump']["original_times"], []
            )

            detx = join(DATA_DIR, 'detx_v3.detx')

            h5group = '/hits'
            cal = Calibration(filename=detx)

            with tb.File(outfile, "a") as f:
                calibrate.initialise_arrays(h5group, f)
                calibrate.calibrate_hits(f, cal, 90, h5group, False)

            with tb.File(outfile, "r") as f:
                for i in range(n_total_hits):
                    channel_id = f.get_node('/hits/channel_id')[i]
                    dom_id = f.get_node('/hits/dom_id')[i]
                    time = f.get_node('/hits/time')[i]
                    t0 = f.get_node('/hits/t0')[i]

                    original_time = original_times[i]

                    pmt = cal.detector.get_pmt(dom_id, channel_id)
                    self.assertAlmostEqual(t0, pmt.t0, 4)
                    self.assertAlmostEqual(t0, time - original_time)
