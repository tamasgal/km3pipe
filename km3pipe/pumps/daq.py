# coding=utf-8
# Filename: daq.py
# pylint: disable=locally-disabled
"""
Pumps for the DAQ data formats.

"""
from __future__ import division, absolute_import, print_function

import struct
from struct import unpack

from km3pipe import Pump, Blob
from km3pipe.logger import get_logger


log = get_logger(__name__)  # pylint: disable=C0103

DATA_TYPES = {
    101: 'DAQSuperFrame',
    201: 'DAQSummaryFrame',
    1001: 'DAQTimeslice',
    2001: 'DAQSummaryslice',
    10001: 'DAQEvent',
}


class DAQPump(Pump):
    """A pump for binary DAQ files."""

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename')
        self.frame_positions = []

        if self.filename:
            self.open_file(self.filename)
            self.determine_frame_positions()
        else:
            log.warn("No filename specified. Take care of the file handling!")

    def next_frame(self):
        """Get the next frame from file"""
        blob_file = self.blob_file
        try:
            preamble = DAQPreamble(from_file=blob_file)
        except struct.error:
            raise StopIteration
        header = DAQHeader(from_file=blob_file)
        raw_data = blob_file.read(preamble.length - DAQPreamble.size - DAQHeader.size)

        blob = Blob()
        blob[DATA_TYPES[preamble.data_type]] = [preamble, header]
        return blob

    def determine_frame_positions(self):
        """Record the file pointer position of each frame"""
        self.rewind_file()
        try:
            while True:
                pointer_position = self.blob_file.tell()
                length = struct.unpack('<i', self.blob_file.read(4))[0]
                self.blob_file.seek(length - 4, 1)
                self.frame_positions.append(pointer_position)
        except struct.error:
            pass
        self.rewind_file()
        print("Found {0} frames.".format(len(self.frame_positions)))

    def process(self, blob):
        """Pump the next blob to the modules"""
        print(self.next_frame())
        return blob

    def finish(self):
        """Clean everything up"""
        self.blob_file.close()





# def foo():
#     print "Preamble:"
#     print struct.unpack('<ii', file.read(8))
#     print "Header:"
#     print struct.unpack('<iiq', file.read(16))
#     print "Subheader:"
#     print struct.unpack('<i', file.read(4))
#     print "Summary frames:"
#     print struct.unpack('i' + 'c'*31, file.read(35))
#     print struct.unpack('i' + 'c'*31, file.read(35))
#     print struct.unpack('i' + 'c'*31, file.read(35))
#
#     print "Header:"
#     print struct.unpack('<iii', file.read(12))
#     print "Timestamp"
#     print struct.unpack('<Q', file.read(8))


class DAQPreamble(object):
    """Wrapper for the JDAQPreamble binary format."""
    size = 8

    def __init__(self, byte_data=None, from_file=None):
        self.length = None
        self.data_type = None
        if byte_data:
            self.parse_byte_data(byte_data)
        if from_file:
            self.parse_file(from_file)

    def parse_byte_data(self, byte_data):
        """Extract the values from byte string"""
        self.length, self.data_type = unpack('<ii', byte_data)

    def parse_file(self, file_obj):
        """Directly read from file handler.

        Note that this will move the file pointer.

        """
        byte_data = file_obj.read(self.size)
        self.parse_byte_data(byte_data)


class DAQHeader(object):
    """Wrapper for the JDAQHeader binary format."""
    size = 16

    def __init__(self, byte_data=None, from_file=None):
        self.run = None
        self.time_slice = None
        self.time_stamp = None
        if byte_data:
            self.parse_byte_data(byte_data)
        if from_file:
            self.parse_file(from_file)

    def parse_byte_data(self, byte_data):
        """Extract the values from byte string"""
        run, time_slice, time_stamp = unpack('<iiQ', byte_data)
        self.run = run
        self.time_slice = time_slice
        self.time_stamp = time_stamp

    def parse_file(self, file_obj):
        """Directly read from file handler.

        Note that this will move the file pointer.

        """
        byte_data = file_obj.read(self.size)
        self.parse_byte_data(byte_data)



class DAQSummarySlice(object):
    pass
