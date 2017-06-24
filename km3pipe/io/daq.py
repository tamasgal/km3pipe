# coding=utf-8
# Filename: daq.py
# pylint: disable=R0903
"""
Pumps for the DAQ data formats.

"""
from __future__ import division, absolute_import, print_function

from collections import defaultdict
from io import BytesIO
import math
import struct
from struct import unpack
import pprint

import numpy as np

from km3pipe.core import Pump, Module, Blob
from km3pipe.dataclasses import EventInfo, HitSeries
from km3pipe.sys import ignored
from km3pipe.logger import logging

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = logging.getLogger(__name__)  # pylint: disable=C0103

DATA_TYPES = {
    101: 'DAQSuperFrame',
    201: 'DAQSummaryFrame',
    1001: 'DAQTimeslice',
    2001: 'DAQSummaryslice',
    10001: 'DAQEvent',
}
MINIMAL_RATE_HZ = 2.0e3
MAXIMAL_RATE_HZ = 2.0e6



class TimesliceParser(Module):
    """Preliminary parser for DAQTimeslice"""
    def process(self, blob):
        if str(blob['CHPrefix'].tag) != 'IO_TSL':
            log.error("Not an IO_TSL blob")
            return

        try:
            data = BytesIO(blob['CHData'])
            tsl_size, datatype = unpack('<ii', data.read(8))
            det_id, run, sqnr = unpack('<iii', data.read(12))
            timestamp, ns_ticks, n_frames = unpack('<iii', data.read(12))

            ts_frames = blob['TimesliceFrames'] = defaultdict(list)

            for i in range(n_frames):
                frame_size, datatype = unpack('<ii', data.read(8))
                det_id, run, sqnr = unpack('<iii', data.read(12))
                timestamp, ns_ticks, dom_id = unpack('<iii', data.read(12))
                dom_status = unpack('<iiiii', data.read(5*4))
                n_hits = unpack('<i', data.read(4))[0]
                hits = []
                for j in range(n_hits):
                    hit = unpack('!BlB', data.read(6))
                    ts_frames[dom_id].append(hit)
        except struct.error:
            log.error("Could not parse Timeslice")
            log.error(blob.keys())
        else:
            return blob


class DAQPump(Pump):
    """A pump for binary DAQ files."""

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename')
        self.frame_positions = []
        self.index = 0

        if self.filename:
            self.open_file(self.filename)
            self.determine_frame_positions()
        else:
            log.warn("No filename specified. Take care of the file handling!")

    def next_blob(self):
        """Get the next frame from file"""
        blob_file = self.blob_file
        try:
            preamble = DAQPreamble(file_obj=blob_file)
        except struct.error:
            raise StopIteration

        try:
            data_type = DATA_TYPES[preamble.data_type]
        except KeyError:
            log.error("Unkown datatype: {0}".format(preamble.data_type))
            data_type = 'Unknown'

        blob = Blob()
        blob[data_type] = None
        blob['DAQPreamble'] = preamble

        if data_type == 'DAQSummaryslice':
            daq_frame = DAQSummaryslice(blob_file)
            blob[data_type] = daq_frame
            blob['DAQHeader'] = daq_frame.header
        elif data_type == 'DAQEvent':
            daq_frame = DAQEvent(blob_file)
            blob[data_type] = daq_frame
            blob['DAQHeader'] = daq_frame.header
        else:
            log.warning("Skipping DAQ frame with data type code '{0}'."
                        .format(preamble.data_type))
            blob_file.seek(preamble.length - DAQPreamble.size, 1)

        return blob

    def seek_to_frame(self, index):
        """Move file pointer to the frame with given index."""
        pointer_position = self.frame_positions[index]
        self.blob_file.seek(pointer_position, 0)

    def get_blob(self, index):
        """Return blob at given index."""
        self.seek_to_frame(index)
        return self.next_blob()

    def determine_frame_positions(self):
        """Record the file pointer position of each frame"""
        self.rewind_file()
        with ignored(struct.error):
            while True:
                pointer_position = self.blob_file.tell()
                length = struct.unpack('<i', self.blob_file.read(4))[0]
                self.blob_file.seek(length - 4, 1)
                self.frame_positions.append(pointer_position)
        self.rewind_file()
        log.info("Found {0} frames.".format(len(self.frame_positions)))

    def process(self, blob):
        """Pump the next blob to the modules"""
        return self.next_blob()

    def finish(self):
        """Clean everything up"""
        self.blob_file.close()

    def __len__(self):
        if not self.frame_positions:
            self.determine_frame_positions()
        return len(self.frame_positions)

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        try:
            blob = self.get_blob(self.index)
        except IndexError:
            self.index = 0
            raise StopIteration
        self.index += 1
        return blob

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_blob(index)
        elif isinstance(index, slice):
            return self._slice_generator(index)
        else:
            raise TypeError("index must be int or slice")

    def _slice_generator(self, index):
        """A simple slice generator for iterations"""
        start, stop, step = index.indices(len(self))
        for i in range(start, stop, step):
            yield self.get_blob(i)


class DAQProcessor(Module):
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.index = 0

    def process(self, blob):
        tag = str(blob['CHPrefix'].tag)
        data = blob['CHData']

        if tag == 'IO_EVT':
            self.process_event(data, blob)
        if tag == 'IO_SUM':
            self.process_summaryslice(data, blob)

        return blob

    def process_event(self, data, blob):
        data_io = BytesIO(data)
        preamble = DAQPreamble(file_obj=data_io)  # noqa
        event = DAQEvent(file_obj=data_io)
        header = event.header

        hits = event.snapshot_hits
        n_hits = event.n_snapshot_hits
        if n_hits == 0:
            return
        dom_ids, channel_ids, times, tots = zip(*hits)
        zeros = np.zeros(n_hits)
        nans = np.full_like(zeros, np.nan)
        triggereds = np.zeros(n_hits)
        triggered_map = {}
        for triggered_hit in event.triggered_hits:
            dom_id, pmt_id, time, tot, _ = triggered_hit
            triggered_map[(dom_id, pmt_id, time, tot)] = True
        for idx, hit in enumerate(hits):
            triggereds[idx] = hit in triggered_map

        hit_series = HitSeries.from_arrays(
            channel_ids,
            nans,  # dir_x
            nans,  # dir_y
            nans,  # dir_z
            dom_ids,
            range(n_hits),  # id
            zeros,  # pmt_id
            nans,  # pos_x
            nans,  # pos_y
            nans,  # pos_z
            zeros,  # t0
            times,
            tots,
            triggereds,
            self.index)

        blob['Hits'] = hit_series

        event_info = EventInfo((
            header.det_id,
            self.index,
            # header.time_slice,
            0,  # livetime_sec
            0, 0,  # MC ID and time
            0, 0,  # n evts/files gen
            event.overlays,
            # header.run,
            event.trigger_counter, event.trigger_mask,
            header.ticks * 16, header.time_stamp,
            0, 0, 0,  # MC weights
            0))
        blob['EventInfo'] = event_info

        self.index += 1

    def process_summaryslice(self, data, blob):
        data_io = BytesIO(data)
        preamble = DAQPreamble(file_obj=data_io)
        summaryslice = DAQSummaryslice(file_obj=data_io)
        blob["RawSummaryslice"] = summaryslice


class DAQPreamble(object):
    """Wrapper for the JDAQPreamble binary format.

    Args:
      file_obj (file): The binary file, where the file pointer is at the
        beginning of the header.

    Attributes:
      size (int): The size of the original DAQ byte representation.
      data_type (int): The data type of the following frame. The coding is
        stored in the ``DATA_TYPES`` dictionary::

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

    def __repr__(self):
        description = "Length: {0}\nDataType: {1}"\
            .format(self.length, self.data_type)
        return description


class DAQHeader(object):
    """Wrapper for the JDAQHeader binary format.

    Args:
      file_obj (file): The binary file, where the file pointer is at the
        beginning of the header.

    Attributes:
      size (int): The size of the original DAQ byte representation.

    """
    size = 20

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
        chunks = unpack('<iiiii', byte_data[:self.size])
        det_id, run, time_slice, time_stamp, ticks = chunks
        self.det_id = det_id
        self.run = run
        self.time_slice = time_slice
        self.time_stamp = time_stamp
        self.ticks = ticks

    def _parse_file(self, file_obj):
        """Directly read from file handler.

        Note:
          This will move the file pointer.

        """
        byte_data = file_obj.read(self.size)
        self._parse_byte_data(byte_data)

    def __repr__(self):
        description = "Run: {0}\nTime slice: {1}\nTime stamp: {2} ({3})"\
                      .format(self.run, self.time_slice, self.time_stamp,
                              self.ticks)
        return description


class DAQSummaryslice(object):
    """Wrapper for the JDAQSummarySlice binary format.

    Args:
      file_obj (file): The binary file, where the file pointer is at the
        beginning of the header.

    Attributes:
      n_summary_frames (int): The number of summary frames.
      summary_frames (dict): The PMT rates for each DOM. The key is the DOM
        identifier and the corresponding value is a sorted list of PMT rates
        in [Hz].
      dom_rates (dict): The overall DOM rate for each DOM.

    """
    def __init__(self, file_obj):
        self.header = DAQHeader(file_obj=file_obj)
        self.n_summary_frames = unpack('<i', file_obj.read(4))[0]
        self.summary_frames = {}
        self.dq_status = {}
        self.dom_status = {}
        self.dom_rates = {}

        self._parse_summary_frames(file_obj)

    def _parse_summary_frames(self, file_obj):
        """Iterate through the byte data and fill the summary_frames"""
        for _ in range(self.n_summary_frames):
            dom_id = unpack('<i', file_obj.read(4))[0]
            dq_status = file_obj.read(4)  # probably dom status? # noqa
            dom_status = unpack('<iiii', file_obj.read(16))
            raw_rates = unpack('b'*31, file_obj.read(31))
            pmt_rates = [self._get_rate(value) for value in raw_rates]
            self.summary_frames[dom_id] = pmt_rates
            self.dq_status[dom_id] = dq_status
            self.dom_status[dom_id] = dom_status
            self.dom_rates[dom_id] = np.sum(pmt_rates)

    def _get_rate(self, value):
        """Return the rate in Hz from the short int value"""
        if value == 0:
            return 0
        else:
            return MINIMAL_RATE_HZ * math.exp(value * self._get_factor())

    def _get_factor(self):
        return math.log(MAXIMAL_RATE_HZ / MINIMAL_RATE_HZ) / 255


class DAQEvent(object):
    """Wrapper for the JDAQEvent binary format.

    Args:
      file_obj (file): The binary file, where the file pointer is at the

    Attributes:
      trigger_counter (int): Incremental identifier of the occurred trigger.
      trigger_mask (int): The trigger type(s) satisfied.
      overlays (int): Number of merged events.
      n_triggered_hits (int): Number of hits satisfying the trigger conditions.
      n_snapshot_hits (int): Number of snapshot hits.
      triggered_hits (list): A list of triggered hits
        (dom_id, pmt_id, tdc_time, tot, (trigger_mask,))
      snapshot_hits (list): A list of snapshot hits
        (dom_id, pmt_id, tdc_time, tot)

    """
    def __init__(self, file_obj):
        self.header = DAQHeader(file_obj=file_obj)
        self.trigger_counter = unpack('<Q', file_obj.read(8))[0]
        self.trigger_mask = unpack('<Q', file_obj.read(8))[0]
        self.overlays = unpack('<i', file_obj.read(4))[0]

        self.n_triggered_hits = unpack('<i', file_obj.read(4))[0]
        self.triggered_hits = []
        self._parse_triggered_hits(file_obj)

        self.n_snapshot_hits = unpack('<i', file_obj.read(4))[0]
        self.snapshot_hits = []
        self._parse_snapshot_hits(file_obj)

    def _parse_triggered_hits(self, file_obj):
        """Parse and store triggered hits."""
        for _ in range(self.n_triggered_hits):
            dom_id, pmt_id = unpack('<ib', file_obj.read(5))
            tdc_time = unpack('>I', file_obj.read(4))[0]
            tot = unpack('<b', file_obj.read(1))[0]
            trigger_mask = unpack('<Q', file_obj.read(8))
            self.triggered_hits.append((dom_id, pmt_id, tdc_time, tot,
                                       trigger_mask))

    def _parse_snapshot_hits(self, file_obj):
        """Parse and store snapshot hits."""
        for _ in range(self.n_snapshot_hits):
            dom_id, pmt_id = unpack('<ib', file_obj.read(5))
            tdc_time = unpack('>I', file_obj.read(4))[0]
            tot = unpack('<b', file_obj.read(1))[0]
            self.snapshot_hits.append((dom_id, pmt_id, tdc_time, tot))

    def __repr__(self):
        string = '\n'.join((
            " Number of triggered hits: " + str(self.n_triggered_hits),
            " Number of snapshot hits: " + str(self.n_snapshot_hits)
        ))
        string += "\nTriggered hits:\n"
        string += pprint.pformat(self.triggered_hits)
        string += "\nSnapshot hits:\n"
        string += pprint.pformat(self.snapshot_hits)
        return string


class TMCHData(object):
    """Monitoring Channel data."""
    def __init__(self, file_obj):
        f = file_obj

        data_type = f.read(4)
        if data_type != b'TMCH':
            raise ValueError("Invalid datatype: {0}".format(data_type))

        self.run = unpack('>I', f.read(4))[0]
        self.sequence_number = unpack('>I', f.read(4))[0]  # not sure
        self.utc_seconds = unpack('>I', f.read(4))[0]
        self.utc_nanoseconds = unpack('>I', f.read(4))[0] * 16
        self.dom_id = unpack('>I', f.read(4))[0]
        self.dom_status_1 = unpack('>I', f.read(4))[0]  # not sure
        self.dom_status_2 = unpack('>I', f.read(4))[0]  # not sure
        self.dom_status_3 = unpack('>I', f.read(4))[0]  # not sure
        self.dom_status_4 = unpack('>I', f.read(4))[0]  # not sure
        self.pmt_rates = [r*10.0 for r in unpack('>' + 31*'I', f.read(31*4))]
        self.pad = unpack('>I', f.read(4))[0]  # not sure
        self.valid = unpack('>I', f.read(4))[0]  # not sure
        self.yaw, self.pitch, self.roll = unpack('>fff', f.read(12))
        self.ax, self.ay, self.az = unpack('>fff', f.read(12))
        self.gx, self.gy, self.gz = unpack('>fff', f.read(12))
        self.hx, self.hy, self.hz = unpack('>fff', f.read(12))
        self.temp = unpack('>H', f.read(2))[0] / 100.0
        self.humidity = unpack('>H', f.read(2))[0] / 100.0
        # self.det_id = unpack('>I', f.read(4))[0]  # not sure
        # self.n_packets = unpack('>H', f.read(2))[0]  # not sure
        # self.highest_packet_number = unpack('>H', f.read(2))[0]  # not sure
        # self.n_items = unpack('>I', f.read(4))[0]  # not sure

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return self.__str__()
