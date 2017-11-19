# coding=utf-8
# Filename: test_calib.py
# pylint: disable=C0111,E1003,R0904,C0103,R0201,C0102
from __future__ import division, absolute_import, print_function

from km3pipe.testing import TestCase, StringIO, MagicMock, patch
from km3pipe.core import Pipeline, Module, Pump, Blob
from km3pipe.calib import Calibration
from km3pipe.dataclasses import HitSeries

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
            cal = Calibration(filename='foo')

    @patch('km3pipe.calib.Detector')
    def test_init_with_filename(self, mock_detector):
        cal = Calibration(filename='foo.detx')
        mock_detector.assert_called_with(filename='foo.detx')

    @patch('km3pipe.calib.Detector')
    def test_init_with_det_id(self, mock_detector):
        cal = Calibration(det_id=1)
        mock_detector.assert_called_with(t0set=None, calibration=None, det_id=1)
        cal = Calibration(det_id=1, calibration=2, t0set=3)
        mock_detector.assert_called_with(t0set=3, calibration=2, det_id=1)

    def test_apply_to_list(self):
        cal = Calibration()
        hits = [1, 2, 3]
        cal._apply_to_hitseries = MagicMock()
        cal.apply(hits)
        cal._apply_to_hitseries.assert_called_with(hits)

    def test_apply_to_hitseries(self):

        class FakeDetector(object):
            def __init__(self):
                self._pmts_by_dom_id = {}
                self._pmts_by_id = {}

            def pmt_with_id(self, i):
                pmt = MagicMock(dir=np.array((i*10+i, i*10+i+1, i*10+i+2)),
                                pos=np.array((i*100+i, i*100+i+1, i*100+i+2)),
                                t0=1000*i)
                return pmt

        cal = Calibration(detector=FakeDetector())

        n = 5
        ids = np.arange(n)
        dom_ids = np.arange(n)
        dir_xs = np.arange(n)
        dir_ys = np.arange(n)
        dir_zs = np.arange(n)
        times = np.arange(n)
        tots = np.arange(n)
        channel_ids = np.arange(n)
        triggereds = np.ones(n)
        pmt_ids = np.arange(n)
        t0s = np.arange(n)
        pos_xs = np.arange(n)
        pos_ys = np.arange(n)
        pos_zs = np.arange(n)

        hits = HitSeries.from_arrays(
            channel_ids,
            dir_xs,
            dir_ys,
            dir_zs,
            dom_ids,
            ids,
            pmt_ids,
            pos_xs,
            pos_ys,
            pos_zs,
            t0s,
            times,
            tots,
            triggereds,
            0,      # event_id
        )

        self.assertEqual(0, hits[0].time)
        self.assertEqual(4, hits[4].time)
        self.assertFalse(np.isnan(hits[2].pos_y))

        cal._apply_to_hitseries(hits)

        self.assertAlmostEqual(303, hits[3].pos_x)
        self.assertAlmostEqual(304, hits[3].pos_y)
        self.assertAlmostEqual(305, hits[3].pos_z)
        self.assertAlmostEqual(406, hits[4].pos_z)
        self.assertAlmostEqual(2, hits[0].dir_z)
        self.assertAlmostEqual(12, hits[1].dir_y)
        self.assertAlmostEqual(22, hits[2].dir_x)

        self.assertEqual(1001, hits[1].time)
        self.assertEqual(4004, hits[4].time)

        for idx, hit in enumerate(hits):
            h = hit
            if idx == 3:
                break

        self.assertAlmostEqual(303.0, h.pos_x)
