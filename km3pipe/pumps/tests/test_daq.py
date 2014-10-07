# coding=utf-8
# Filename: test_daq.py
"""
...

"""
from __future__ import division, absolute_import, print_function

from km3pipe.testing import *
from km3pipe.pumps.daq import DAQPump, DAQPreamble, DAQHeader

import binascii


HEX_DATA = ''.join("85000000d1070000ae01000001000000000000000000000003000000" +
                   "65000000000000000000000000000000000000000000000000000000" +
                   "00000000000000660000000000000000000000000000000000000000" +
                   "00000000000000000000000000006700000000000000000000000000" +
                   "00000000000000000000000000000000000000000070000000112700" +
                   "00ae0100000200000000000000000080000000000000000000020000" +
                   "000000000000000000020000006700000007f5f40500306700000012" +
                   "f8f405002704000000660000000a6df50500186700000007f5f40500" +
                   "306700000012f8f4050027670000000c1bf605002166000000112700" +
                   "00ae0100000200000000000000000080000100000000000000020000" +
                   "0000000000000000000200000067000000108b1e0b00266700000011" +
                   "901e0b0012030000006500000016f41f0b002467000000108b1e0b00" +
                   "266700000011901e0b00125c00000011270000ae0100000200000000" +
                   "00000000008000020000000000000002000000000000000000000002" +
                   "00000066000000088b950f0023660000001884950f001a0200000066" +
                   "0000001884950f001a66000000088b950f00235c00000011270000ae" +
                   "01000002000000000000000000800003000000000000000200000000" +
                   "000000000000000200000065000000040a1a130022650000000a0e1a" +
                   "1300280200000065000000040a1a130022650000000a0e1a130028")
BINARY_DATA = binascii.unhexlify(HEX_DATA)
TEST_FILE = StringIO(BINARY_DATA)


class TestDAQPump(TestCase):

    def setUp(self):
        TEST_FILE.seek(0, 0)
        self.pump = DAQPump()
        self.pump.blob_file = TEST_FILE

    def test_determine_frame_positions(self):
        pump = self.pump
        pump.determine_frame_positions()
        self.assertListEqual([0, 133, 245, 347, 439], pump.frame_positions)

    def test_next_frame_finds_correct_frame_types(self):
        pump = self.pump
        blob = pump.next_frame()
        self.assertTrue(blob.has_key('DAQSummaryslice'))
        blob = pump.next_frame()
        self.assertTrue(blob.has_key('DAQEvent'))
        blob = pump.next_frame()
        self.assertTrue(blob.has_key('DAQEvent'))
        blob = pump.next_frame()
        self.assertTrue(blob.has_key('DAQEvent'))
        blob = pump.next_frame()
        self.assertTrue(blob.has_key('DAQEvent'))

    def test_next_frame_raises_stop_iteration_when_eof_reached(self):
        pump = self.pump
        with self.assertRaises(StopIteration):
            for i in range(6):
                pump.next_frame()


class TestDAQPreamble(TestCase):

    def setUp(self):
        TEST_FILE.seek(0, 0)

    def test_init_with_byte_data(self):
        byte_data = binascii.unhexlify("85000000D1070000")
        preamble = DAQPreamble(byte_data=byte_data)
        self.assertEqual(133, preamble.length)
        self.assertEqual(2001, preamble.data_type)

    def test_parse_from_file(self):
        self.setUp()
        preamble = DAQPreamble(from_file=TEST_FILE)
        self.assertEqual(133, preamble.length)
        self.assertEqual(2001, preamble.data_type)



class TestDAQHeader(TestCase):

    def test_init_with_byte_data(self):
        byte_data = binascii.unhexlify("AE010000010000000000000000000000")
        header = DAQHeader(byte_data=byte_data)
        self.assertEqual(430, header.run)
        self.assertEqual(1, header.time_slice)

    def test_parse_from_file(self):
        self.setUp()
        header = DAQHeader(from_file=TEST_FILE)
        self.assertEqual(430, header.run)
        self.assertEqual(1, header.time_slice)