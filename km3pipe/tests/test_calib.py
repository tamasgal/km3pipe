# Filename: test_calib.py
# pylint: disable=C0111,E1003,R0904,C0103,R0201,C0102

from km3pipe.dataclasses import Table
from km3pipe.hardware import Detector
from km3pipe.testing import TestCase, MagicMock, patch, skip
from km3pipe.calib import Calibration

from .test_hardware import EXAMPLE_DETX

import numpy as np

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
        det = Detector()
        det._det_file = EXAMPLE_DETX
        det._parse_header()
        Calibration(detector=det)

    def test_apply_to_hits_with_pmt_id(self):
        det = Detector()
        det._det_file = EXAMPLE_DETX
        det._parse_header()
        det._parse_doms()
        calib = Calibration(detector=det)

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
        det = Detector()
        det._det_file = EXAMPLE_DETX
        det._parse_header()
        det._parse_doms()
        calib = Calibration(detector=det)

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
        det = Detector()
        det._det_file = EXAMPLE_DETX
        det._parse_header()
        det._parse_doms()
        calib = Calibration(detector=det)
        c_tshits = calib.apply(tshits)
        assert len(c_tshits) == len(tshits)
        assert np.allclose([40, 80, 90], c_tshits.t0)
        # TimesliceHits is using int4 for times, so it's truncated when we pass in float64
        assert np.allclose([50.0, 91.0, 102.0], c_tshits.time)
