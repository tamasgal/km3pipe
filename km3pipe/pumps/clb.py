# coding=utf-8
# Filename: clb.py
# pylint: disable=locally-disabled
"""
Pumps for the CLB data formats.

"""
from __future__ import division, absolute_import, print_function

import struct
from struct import unpack
import binascii
from collections import namedtuple
import datetime

from km3pipe import Pump, Blob
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103


class CLBPump(Pump):
    """A pump for binary CLB files."""

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename')
        self.packet_positions = []
        self.index = 0

        if self.filename:
            self.open_file(self.filename)
            self.determine_packet_positions()
        else:
            log.warn("No filename specified. Take care of the file handling!")

    def determine_packet_positions(self):
        """Record the file pointer position of each frame"""
        self.rewind_file()
        try:
            while True:
                pointer_position = self.blob_file.tell()
                length = struct.unpack('<i', self.blob_file.read(4))[0]
                self.packet_positions.append(pointer_position)
                self.blob_file.seek(length, 1)
        except struct.error:
            pass
        self.rewind_file()
        print("Found {0} CLB UDP packets.".format(len(self.packet_positions)))

    def seek_to_packet(self, index):
        """Move file pointer to the packet with given index."""
        pointer_position = self.packet_positions[index]
        self.blob_file.seek(pointer_position, 0)

    def next_blob(self):
        try:
            length = struct.unpack('<i', self.blob_file.read(4))[0]
        except struct.error:
            raise StopIteration
        header = CLBHeader(file_obj=self.blob_file)
        blob = {'CLBHeader': header}
        remaining_length = length - header.size
        pmt_data = []
        for _ in xrange(int(remaining_length/6)):
            channel_id, timestamp, tot = struct.unpack('>cic',
                                                       self.blob_file.read(6))
            pmt_data.append(PMTData(ord(channel_id), timestamp, ord(tot)))
        blob['PMTData'] = pmt_data
        return blob

    def get_blob(self, index):
        """Return blob at given index."""
        self.seek_to_packet(index)
        return self.next_blob()

    def process(self, blob):
        blob = self.next_blob()
        return blob


class CLBHeader(object):
    """Wrapper for the CLB Common Header binary format.

    Args:
      file_obj (file): The binary file, where the file pointer is at the
        beginning of the header.

    Attributes:
      size (int): The size of the original DAQ byte representation.

    """
    size = 28

    def __init__(self, byte_data=None, file_obj=None):
        self.data_type = None
        self.run_number = None
        self.udp_sequence = None
        self.timestamp = None
        self.ns_ticks = None
        self.human_readable_timestamp = None
        self.dom_id = None
        self.dom_status = None
        self.time_valid = None
        if byte_data:
            self._parse_byte_data(byte_data)
        if file_obj:
            self._parse_file(file_obj)

    def __str__(self):
        description = ("CLBHeader\n"
                       "    Data type:    {self.data_type}\n"
                       "    Run number:   {self.run_number}\n"
                       "    UDP sequence: {self.udp_sequence}\n"
                       "    Time stamp:   {self.timestamp}\n"
                       "                  {self.human_readable_timestamp}\n"
                       "    Ticks [ns]:   {self.ns_ticks}\n"
                       "    DOM ID:       {self.dom_id}\n"
                       "    DOM status:   {self.dom_status}\n"
                       "".format(self=self))
        return description

    def _parse_byte_data(self, byte_data):
        """Extract the values from byte string."""
        self.data_type = ''.join(unpack('cccc', byte_data[:4]))
        self.run_number = unpack('>i', byte_data[4:8])[0]
        self.udp_sequence = unpack('>i', byte_data[8:12])[0]
        self.timestamp, self.ns_ticks = unpack('>II', byte_data[12:20])
        self.dom_id = binascii.hexlify(''.join(unpack('cccc', byte_data[20:24])))

        b = unpack('>I', byte_data[24:28])[0]
        #first_bit = b >> 7
        #self.time_valid = bool(first_bit)
        self.dom_status = "{0:032b}".format(b)

        self.human_readable_timestamp = datetime.datetime.fromtimestamp(
            int(self.timestamp)).strftime('%Y-%m-%d %H:%M:%S')



    def _parse_file(self, file_obj):
        """Directly read from file handler.

        Note:
          This will move the file pointer.

        """
        byte_data = file_obj.read(self.size)
        self._parse_byte_data(byte_data)

PMTData = namedtuple('PMTData', 'channel_id timestamp tot')