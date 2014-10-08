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
            preamble = DAQPreamble(file_obj=blob_file)
        except struct.error:
            raise StopIteration

        data_type = DATA_TYPES[preamble.data_type]

        blob = Blob()
        blob[data_type] = None

        if data_type == 'DAQSummaryslice':
            daq_frame = DAQSummarySlice(blob_file)
            blob[data_type] = daq_frame
        else:
            blob_file.seek(preamble.length - DAQPreamble.size, 1)

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


class DAQPreamble(object):
    """Wrapper for the JDAQPreamble binary format.

    Args:
      file_obj (file): The binary file, where the file pointer is at the
        beginning of the header.

    Attributes:
      size (int): The size of the original DAQ byte representation.
      data_type (int): The data type of the following frame. The coding is
        stored in the ``DATA_TYPES`` dictionary:

            101: 'DAQSuperFrame'
            201: 'DAQSummaryFrame'
            1001: 'DAQTimeslice'
            2001: 'DAQSummaryslice'
            10001: 'DAQEvent'

    """
    size = 8

    def __init__(self, byte_data=None, file_obj=None):
        self.length = None
        self.data_type = None
        if byte_data:
            self._parse_byte_data(byte_data)
        if file_obj:
            self._parse_file(file_obj)

    def _parse_byte_data(self, byte_data):
        """Extract the values from byte string."""
        self.length, self.data_type = unpack('<ii', byte_data[:self.size])

    def _parse_file(self, file_obj):
        """Directly read from file handler.

        Note that this will move the file pointer.

        """
        byte_data = file_obj.read(self.size)
        self._parse_byte_data(byte_data)


class DAQHeader(object):
    """Wrapper for the JDAQHeader binary format.

    Args:
      file_obj (file): The binary file, where the file pointer is at the
        beginning of the header.

    Attributes:
      size (int): The size of the original DAQ byte representation.

    """
    size = 16

    def __init__(self, byte_data=None, file_obj=None):
        self.run = None
        self.time_slice = None
        self.time_stamp = None
        if byte_data:
            self._parse_byte_data(byte_data)
        if file_obj:
            self._parse_file(file_obj)

    def _parse_byte_data(self, byte_data):
        """Extract the values from byte string."""
        run, time_slice, time_stamp = unpack('<iiQ', byte_data[:self.size])
        self.run = run
        self.time_slice = time_slice
        self.time_stamp = time_stamp

    def _parse_file(self, file_obj):
        """Directly read from file handler.

        Note:
          This will move the file pointer.

        """
        byte_data = file_obj.read(self.size)
        self._parse_byte_data(byte_data)



class DAQSummarySlice(object):
    """Wrapper for the JDAQSummarySlice binary format.

    Args:
      file_obj (file): The binary file, where the file pointer is at the
        beginning of the header.

    Attributes:
      n_summary_frames (int): The number of summary frames.
      summary_frames (dict): The PMT rates for each DOM. The key is the DOM
        identifier and the corresponding value is a sorted list of PMT rates.

    """
    def __init__(self, file_obj):
        self.header = DAQHeader(file_obj=file_obj)
        self.n_summary_frames = unpack('<i', file_obj.read(4))[0]
        self.summary_frames = {}

        self._parse_summary_frames(file_obj)

    def _parse_summary_frames(self, file_obj):
        """Iterate through the byte data and fill the summary_frames"""
        for i in range(self.n_summary_frames):
            print("frame: {0}".format(i))
            dom_id = unpack('<i', file_obj.read(4))[0]
            pmt_rates = unpack('c'*31, file_obj.read(31))
            self.summary_frames[dom_id] = pmt_rates


class DAQEvent(object):
    """Wrapper for the JDAQEvent binary format.

    Args:
      file_obj (file): The binary file, where the file pointer is at the

    Attributes:
      trigger_counter (int): Incremental identifier of the occurred trigger.
      trigger_mask (int): The trigger type(s) satisfied.
      overlays (int): Number of merged events.
      n_trig_hits (int): Number of hits satisfying the trigger conditions.

    """
    def __init__(self, file_obj):
        self.header = DAQHeader(file_obj=file_obj)
        self.trigger_counter, self.trigger_mask = unpack('<QQ', file_obj.read(16))
        self.overlays, self.n_trig_hits = unpack('<ii', file_obj.read(8))
        self.triggered_hits = []
        self.snapshot_hits = []
        #self._parse_triggered_hits(byte_data[24:self.n_trig_hits*18])
