# coding=utf-8
# Filename: clb.py
"""
Pumps for the CLB data formats.

"""
from __future__ import division, absolute_import, print_function

import struct
from struct import unpack
from binascii import hexlify
from collections import namedtuple
import datetime
import pytz

from km3pipe import Pump
from km3pipe.tools import ignored
from km3pipe.logger import logging

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = logging.getLogger(__name__)  # pylint: disable=C0103


UTC_TZ = pytz.timezone('UTC')


class CLBPump(Pump):
    """A pump for binary CLB files."""

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename')
        self.cache_enabled = self.get('cache_enabled') or False
        self.packet_positions = []
        self.index = 0

        if self.filename:
            self.open_file(self.filename)
            if self.cache_enabled:
                self.determine_packet_positions()

    def determine_packet_positions(self):
        """Record the file pointer position of each frame"""
        print("Analysing file...")
        self.rewind_file()
        with ignored(struct.error):
            while True:
                pointer_position = self.blob_file.tell()
                length = struct.unpack('<i', self.blob_file.read(4))[0]
                self.packet_positions.append(pointer_position)
                self.blob_file.seek(length, 1)
        self.rewind_file()
        print("Found {0} CLB UDP packets.".format(len(self.packet_positions)))

    def seek_to_packet(self, index):
        """Move file pointer to the packet with given index."""
        pointer_position = self.packet_positions[index]
        self.blob_file.seek(pointer_position, 0)

    def next_blob(self):
        """Generate next blob in file"""
        try:
            length = struct.unpack('<i', self.blob_file.read(4))[0]
        except struct.error:
            raise StopIteration
        header = CLBHeader(file_obj=self.blob_file)
        blob = {'CLBHeader': header}
        remaining_length = length - header.size
        pmt_data = []
        for _ in range(int(remaining_length/6)):
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

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        return self.next_blob()

    def finish(self):
        """Clean everything up"""
        self.blob_file.close()


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
        # pylint: disable=E1124
        description = ("CLBHeader\n"
                       "    Data type:    {self.data_type}\n"
                       "    Run number:   {self.run_number}\n"
                       "    UDP sequence: {self.udp_sequence}\n"
                       "    Time stamp:   {self.timestamp}\n"
                       "                  {self.human_readable_timestamp}\n"
                       "    Ticks [16ns]: {self.ns_ticks}\n"
                       "    DOM ID:       {self.dom_id}\n"
                       "    DOM status:   {self.dom_status}\n"
                       "".format(self=self))
        return description

    def __insp__(self):
        return self.__str__()

    def _parse_byte_data(self, byte_data):
        """Extract the values from byte string."""
        self.data_type = b''.join(unpack('cccc', byte_data[:4])).decode()
        self.run_number = unpack('>i', byte_data[4:8])[0]
        self.udp_sequence = unpack('>i', byte_data[8:12])[0]
        self.timestamp, self.ns_ticks = unpack('>II', byte_data[12:20])
        self.dom_id = hexlify(b''.join(unpack('cccc',
                                              byte_data[20:24]))).decode()

        dom_status_bits = unpack('>I', byte_data[24:28])[0]
        self.dom_status = "{0:032b}".format(dom_status_bits)

        print("Timestamp: {0}".format(self.timestamp))
        self.human_readable_timestamp = datetime.datetime.fromtimestamp(
            int(self.timestamp), UTC_TZ).strftime('%Y-%m-%d %H:%M:%S')

    def _parse_file(self, file_obj):
        """Directly read from file handler.

        Note:
          This will move the file pointer.

        """
        byte_data = file_obj.read(self.size)
        self._parse_byte_data(byte_data)

PMTData = namedtuple('PMTData', 'channel_id timestamp tot')
