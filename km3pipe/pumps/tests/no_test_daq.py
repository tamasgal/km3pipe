# coding=utf-8
# Filename: no_test_daq.py
# pylint: disable=C0111,R0904,C0103
"""
...

"""
from __future__ import division, absolute_import, print_function

from km3pipe.testing import TestCase, StringIO, skipIf
from km3pipe.pumps.daq import (DAQPreamble, DAQHeader, DAQSummaryslice)

from binascii import unhexlify


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
BINARY_DATA = unhexlify(HEX_DATA.encode())
try:
    TEST_FILE = StringIO(BINARY_DATA)
except TypeError:
    from io import BytesIO
    TEST_FILE = BytesIO(BINARY_DATA)


class TestDAQPump(TestCase):

    # @skipIf(True, "Needs to clarify final data format specs.")
    # def setUp(self):
    #     TEST_FILE.seek(0, 0)
    #     self.pump = DAQPump()
    #     self.pump.blob_file = TEST_FILE

    @skipIf(True, "Needs to clarify final data format specs.")
    def test_determine_frame_positions(self):
        pump = self.pump
        pump.determine_frame_positions()
        self.assertListEqual([0, 133, 245, 347, 439], pump.frame_positions)

    @skipIf(True, "Needs to clarify final data format specs.")
    def test_next_blob_finds_correct_frame_types(self):
        pump = self.pump
        blob = pump.next_blob()
        self.assertTrue('DAQSummaryslice' in blob)
        blob = pump.next_blob()
        self.assertTrue('DAQEvent' in blob)
        blob = pump.next_blob()
        self.assertTrue('DAQEvent' in blob)
        blob = pump.next_blob()
        self.assertTrue('DAQEvent' in blob)
        blob = pump.next_blob()
        self.assertTrue('DAQEvent' in blob)

    @skipIf(True, "Needs to clarify final data format specs.")
    def test_next_blob_raises_stop_iteration_when_eof_reached(self):
        pump = self.pump
        with self.assertRaises(StopIteration):
            for i in range(6):
                pump.next_blob()

    @skipIf(True, "Needs to clarify final data format specs.")
    def test_seek_to_frame(self):
        pump = self.pump
        pump.determine_frame_positions()
        pump.seek_to_frame(2)
        self.assertEqual(245, pump.blob_file.tell())

    @skipIf(True, "Needs to clarify final data format specs.")
    def test_next_blob_returns_correct_frame_after_seek_to_frame(self):
        pump = self.pump
        pump.determine_frame_positions()
        pump.seek_to_frame(1)
        blob = pump.next_blob()
        self.assertEqual(4, blob['DAQEvent'].n_snapshot_hits)

    @skipIf(True, "Needs to clarify final data format specs.")
    def test_get_blob(self):
        pump = self.pump
        pump.determine_frame_positions()
        blob = pump.get_blob(2)
        self.assertEqual(3, blob['DAQEvent'].n_snapshot_hits)


class TestDAQPreamble(TestCase):

    # def setUp(self):
    #     TEST_FILE.seek(0, 0)

    @skipIf(True, "Needs to clarify final data format specs.")
    def test_init_with_byte_data(self):
        byte_data = unhexlify("85000000D1070000".encode())
        preamble = DAQPreamble(byte_data=byte_data)
        self.assertEqual(133, preamble.length)
        self.assertEqual(2001, preamble.data_type)

    @skipIf(True, "Needs to clarify final data format specs.")
    def test_parse_from_file(self):
        self.setUp()
        preamble = DAQPreamble(file_obj=TEST_FILE)
        self.assertEqual(133, preamble.length)
        self.assertEqual(2001, preamble.data_type)


class TestDAQHeader(TestCase):

    @skipIf(True, "Needs to clarify final data format specs.")
    def test_init_with_byte_data(self):
        byte_data = unhexlify("AE010000010000000000000000000000".encode())
        header = DAQHeader(byte_data=byte_data)
        self.assertEqual(430, header.run)
        self.assertEqual(1, header.time_slice)

    @skipIf(True, "Needs to clarify final data format specs.")
    def test_parse_from_file(self):
        TEST_FILE.seek(8, 0)  # skip preamble
        self.setUp()
        header = DAQHeader(file_obj=TEST_FILE)
        self.assertEqual(430, header.run)
        self.assertEqual(1, header.time_slice)


class TestDAQSummaryslice(TestCase):

    @skipIf(True, "Needs to clarify final data format specs.")
    def test_init_with_a_slice(self):
        TEST_FILE.seek(0, 0)
        DAQPreamble(file_obj=TEST_FILE)
        sum_slice = DAQSummaryslice(TEST_FILE)
        self.assertEqual(3, sum_slice.n_summary_frames)
        print(sum_slice.summary_frames.keys())
        self.assertListEqual([101, 102, 103],
                             list(sum_slice.summary_frames.keys()))
        self.assertEqual(31, len(sum_slice.summary_frames[101]))
        self.assertEqual(31, len(sum_slice.summary_frames[102]))
        self.assertEqual(31, len(sum_slice.summary_frames[103]))


class TestDAQEvent(TestCase):

    # def setUp(self):
    #    TEST_FILE.seek(245, 0)
    #    DAQPreamble(file_obj=TEST_FILE)
    #    self.event = DAQEvent(TEST_FILE)

    @skipIf(True, "Needs to clarify final data format specs.")
    def test_init_with_a_frame(self):
        event = self.event
        self.assertEqual(1, event.trigger_counter)
        self.assertEqual(2, event.trigger_mask)
        self.assertEqual(0, event.overlays)
        self.assertEqual(2, event.n_triggered_hits)

    @skipIf(True, "Needs to clarify final data format specs.")
    def test_triggered_hits(self):
        event = self.event
        self.assertEqual(2, len(event.triggered_hits))
        self.assertTupleEqual((103, 16, 728715, 38), event.triggered_hits[0])
        self.assertTupleEqual((103, 17, 728720, 18), event.triggered_hits[1])

    @skipIf(True, "Needs to clarify final data format specs.")
    def test_snapshot_hits(self):
        event = self.event
        self.assertEqual(3, event.n_snapshot_hits)
        self.assertEqual(3, len(event.snapshot_hits))
        self.assertTupleEqual((101, 22, 729076, 36), event.snapshot_hits[0])
        self.assertTupleEqual((103, 16, 728715, 38), event.snapshot_hits[1])
        self.assertTupleEqual((103, 17, 728720, 18), event.snapshot_hits[2])
