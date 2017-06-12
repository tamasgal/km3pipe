# coding=utf-8
# Filename: test_cmd.py
# pylint: disable=C0111,E1003,R0904,C0103,R0201,C0102
from __future__ import division, absolute_import, print_function

from km3pipe.testing import TestCase, MagicMock, patch
from km3pipe.cmd import detx, update_km3pipe

#from mock import PropertyMock

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


KM3PIPE_GIT = "http://git.km3net.de/km3py/km3pipe.git"


class TestDetx(TestCase):
    @patch('km3pipe.cmd.Detector')
    def test_detector_called_with_correct_det_id(self, mock_detector):
        det = mock_detector.return_value
        det.n_doms = 0
        detx(1)
        mock_detector.assert_called_with(t0set='', det_id=1, calibration='')

    @patch('km3pipe.cmd.Detector')
    def test_detector_write_called_with_correct_filename(self, mock_detector):
        det = mock_detector.return_value
        det.n_doms = 1
        detx(1)
        self.assertTrue(det.write.call_args[0][0]
                        .startswith("KM3NeT_00000001_"))
        self.assertTrue(det.write.call_args[0][0]
                        .endswith(".detx"))

    @patch('km3pipe.cmd.Detector')
    def test_detector_called_with_correct_args(self, mock_detector):
        det = mock_detector.return_value
        det.n_doms = 1
        detx(1, 2, 3)
        mock_detector.assert_called_with(t0set=3, det_id=1, calibration=2)


class TestUpdateKm3pipe(TestCase):
    @patch('km3pipe.cmd.os')
    def test_update_without_args_updates_master(self, mock_os):
        update_km3pipe()
        expected = "pip install -U git+{0}@master".format(KM3PIPE_GIT)
        mock_os.system.assert_called_with(expected)
        update_km3pipe('')
        expected = "pip install -U git+{0}@master".format(KM3PIPE_GIT)
        mock_os.system.assert_called_with(expected)
        update_km3pipe(None)
        expected = "pip install -U git+{0}@master".format(KM3PIPE_GIT)
        mock_os.system.assert_called_with(expected)

    @patch('km3pipe.cmd.os')
    def test_update_branch(self, mock_os):
        branch = 'foo'
        update_km3pipe(branch)
        expected = "pip install -U git+{0}@{1}".format(KM3PIPE_GIT, branch)
        mock_os.system.assert_called_with(expected)
