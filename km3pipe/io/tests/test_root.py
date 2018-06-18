# Filename: test_root.py
# pylint: disable=locally-disabled,C0111,R0904,C0301,C0103,W0212

from km3pipe.testing import TestCase, patch, Mock, surrogate
from km3pipe.io.root import (
    open_rfile, get_hist, get_hist2d, get_hist3d, interpol_hist2d
)

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2018, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestOpenRfile(TestCase):
    @surrogate('rootpy.io.root_open')
    @patch('rootpy.io.root_open')
    def test_call(self, root_open_mock):
        open_rfile('a.root')
        root_open_mock.assert_called_with('a.root', mode='r')


class TestGetHist(TestCase):
    @surrogate('root_numpy.hist2array')
    @patch('km3pipe.io.root.open_rfile')
    @patch('root_numpy.hist2array')
    def test_call(self, hist2array_mock, open_rfile_mock):
        get_hist('a.root', 'histname')
        hist2array_mock.assert_called_once()


class TestGetHist2d(TestCase):
    @surrogate('root_numpy.hist2array')
    @patch('km3pipe.io.root.open_rfile')
    @patch('root_numpy.hist2array')
    def test_call(self, hist2array_mock, open_rfile_mock):
        get_hist2d('a.root', 'histname')
        hist2array_mock.assert_called_once()


class TestGetHist3d(TestCase):
    @surrogate('root_numpy.hist2array')
    @patch('km3pipe.io.root.open_rfile')
    @patch('root_numpy.hist2array')
    def test_call(self, hist2array_mock, open_rfile_mock):
        get_hist3d('a.root', 'histname')
        hist2array_mock.assert_called_once()
