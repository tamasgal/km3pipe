# Filename: test_jpp.py
# pylint: disable=locally-disabled,C0111,R0904,C0301,C0103,W0212
import os
import pathlib
from tempfile import NamedTemporaryFile, gettempdir

from km3pipe.testing import TestCase, surrogate, patch

from km3pipe.io.jpp import (EventPump, TimeslicePump, SummaryslicePump,
                            FitPump, JppCMDWrapper)

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2018, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestEventPump(TestCase):
    @surrogate('jppy.PyJDAQEventReader')
    @patch('jppy.PyJDAQEventReader')
    def test_init(self, reader_mock):
        filename = 'a.root'
        EventPump(filename)
        reader_mock.assert_called_with(filename.encode())

    @surrogate('jppy.PyJDAQEventReader')
    @patch('jppy.PyJDAQEventReader')
    def test_resize_buffers(self, reader_mock):
        filename = 'a.root'
        pump = EventPump(filename)
        assert len(pump._channel_ids) == pump.buf_size
        new_buf_size = 8000
        pump._resize_buffers(new_buf_size)
        assert pump.buf_size == new_buf_size
        assert len(pump._channel_ids) == new_buf_size


class TestTimeslicePump(TestCase):
    @surrogate('jppy.daqtimeslicereader.PyJDAQTimesliceReader')
    @patch('jppy.daqtimeslicereader.PyJDAQTimesliceReader')
    def test_init(self, reader_mock):
        filename = 'a.root'
        TimeslicePump(filename)
        reader_mock.assert_called_with(filename.encode(), b'JDAQTimesliceL1')

    @surrogate('jppy.daqtimeslicereader.PyJDAQTimesliceReader')
    @patch('jppy.daqtimeslicereader.PyJDAQTimesliceReader')
    def test_init_with_specific_stream(self, reader_mock):
        filename = 'a.root'
        TimeslicePump(filename, stream='L0')
        reader_mock.assert_called_with(filename.encode(), b'JDAQTimesliceL0')
        TimeslicePump(filename, stream='SN')
        reader_mock.assert_called_with(filename.encode(), b'JDAQTimesliceSN')

    @surrogate('jppy.daqtimeslicereader.PyJDAQTimesliceReader')
    @patch('jppy.daqtimeslicereader.PyJDAQTimesliceReader')
    def test_resize_buffers(self, reader_mock):
        filename = 'a.root'
        pump = TimeslicePump(filename)
        assert len(pump._channel_ids) == pump.buf_size
        new_buf_size = 8000
        pump._resize_buffers(new_buf_size)
        assert pump.buf_size == new_buf_size
        assert len(pump._channel_ids) == new_buf_size


class TestSummaryslicePump(TestCase):
    @surrogate('jppy.daqsummaryslicereader.PyJDAQSummarysliceReader')
    @patch('jppy.daqsummaryslicereader.PyJDAQSummarysliceReader')
    def test_init(self, reader_mock):
        filename = 'a.root'
        SummaryslicePump(filename)
        reader_mock.assert_called_with(filename.encode())


class TestFitPump(TestCase):
    @surrogate('jppy.PyJFitReader')
    @patch('jppy.PyJFitReader')
    def test_init(self, reader_mock):
        filename = 'a.root'
        FitPump(filename)
        reader_mock.assert_called_with(filename.encode())

    @surrogate('jppy.PyJFitReader')
    @patch('jppy.PyJFitReader')
    def test_resize_buffers(self, reader_mock):
        filename = 'a.root'
        pump = FitPump(filename)
        assert len(pump._pos_xs) == pump.buf_size
        new_buf_size = 8000
        pump._resize_buffers(new_buf_size)
        assert pump.buf_size == new_buf_size
        assert len(pump._pos_xs) == new_buf_size


class TestJppCMDWrapper(TestCase):
    def test_init_without_dir(self):
        JppCMDWrapper()

    def test_init(self):
        bin_dir = os.path.join(gettempdir(), 'jpp_bin')
        os.environ['JPP_BIN'] = bin_dir
        pathlib.Path(bin_dir).mkdir(parents=True, exist_ok=True) 
        tools = [NamedTemporaryFile(prefix='J', dir=bin_dir) for f in range(5)]

        jppw = JppCMDWrapper()
        for tool in tools:
            assert os.path.basename(tool.name) in jppw._tools.keys()
            assert tool.name in jppw._tools.values()

