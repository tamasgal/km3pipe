#!/usr/bin/env python
# coding=utf-8
# Filename: jpp.py
# pylint: disable=
"""
Pump for the jpp file read through aanet interface.

"""
from __future__ import division, absolute_import, print_function

import numpy as np

from km3pipe.core import Pump, Blob
from km3pipe.dataclasses import (EventInfo, TimesliceFrameInfo,
                                 SummaryframeInfo, HitSeries,
                                 TimesliceHitSeries, RawHitSeries)
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Thomas Heid"]
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class JPPPump(Pump):
    """A pump for JPP ROOT files.

    This pump will be replaced soon by ``EventPump``, ``TimeslicePump`` and
    ``SummaryslicePump``.

    Parameters
    ----------
    filename: str
        Name of the file to open.
    index: int
        The number to start the event indexing with [default: 0]
    with_summaryslices: bool
        Extract summary slices [default: False]
    with_timeslice_hits: bool
        Extract the hits from the timeslices.
        This can make the file huge. [default: False]
    """

    def configure(self):

        try:
            import jppy  # noqa
        except ImportError:
            raise ImportError("\nEither Jpp or jppy could not be found."
                              "\nMake sure you source the JPP environmanet "
                              "and have jppy installed")

        self.event_index = self.get('index') or 0
        self.with_summaryslices = self.get('with_summaryslices') or False
        self.with_timeslice_hits = self.get('with_timeslice_hits') or False
        self.timeslice_index = 0
        self.timeslice_frame_index = 0
        self.summaryslice_index = 0
        self.summaryslice_frame_index = 0
        self.filename = self.get('filename')

        self.event_reader = jppy.PyJDAQEventReader(self.filename)
        self.timeslice_reader = jppy.PyJDAQTimesliceReader(self.filename)
        self.summaryslice_reader = jppy.PyJDAQSummarysliceReader(self.filename)
        self.blobs = self.blob_generator()

    def blob_generator(self):
        while self.with_timeslice_hits and self.timeslice_reader.has_next:
            self.timeslice_frame_index = 0
            self.timeslice_reader.retrieve_next_timeslice()
            while self.timeslice_reader.has_next_superframe:
                try:
                    yield self.extract_timeslice_frame()
                except IndexError:
                    log.warning("Skipping broken frame.")
                else:
                    self.timeslice_frame_index += 1
                finally:
                    self.timeslice_reader.retrieve_next_superframe()
            self.timeslice_index += 1

        while self.with_summaryslices and self.summaryslice_reader.has_next:
            self.summaryslice_frame_index = 0
            self.summaryslice_reader.retrieve_next_summaryslice()
            while self.summaryslice_reader.has_next_frame:
                yield self.extract_summaryslice_frame()
                self.summaryslice_reader.retrieve_next_frame()
                self.summaryslice_frame_index += 1
            self.summaryslice_index += 1

        while self.event_reader.has_next:
            try:
                yield self.extract_event()
            except IndexError:
                pass

        raise StopIteration

    def extract_event(self):
        blob = Blob()
        r = self.event_reader
        r.retrieve_next_event()  # do it at the beginning!

        n = r.number_of_snapshot_hits
        channel_ids = np.zeros(n, dtype='i')
        dom_ids = np.zeros(n, dtype='i')
        times = np.zeros(n, dtype='i')
        tots = np.zeros(n, dtype='i')
        triggereds = np.zeros(n, dtype='i')

        r.get_hits(channel_ids, dom_ids, times, tots, triggereds)

        nans = np.full(n, np.nan, dtype='<f8')
        hit_series = HitSeries.from_arrays(
            channel_ids, nans, nans, nans, dom_ids, np.arange(n), np.zeros(n),
            nans, nans, nans, nans, times, tots, triggereds, self.event_index
        )

        event_info = EventInfo(np.array((
            r.det_id, r.frame_index,
            0,  # livetime_sec
            0, 0,  # MC ID and time
            0,  # n_events_gen
            0,  # n_files_gen
            r.overlays,
            r.trigger_counter, r.trigger_mask,
            r.utc_nanoseconds, r.utc_seconds,
            np.nan, np.nan, np.nan,   # w1-w3
            0,  # run
            self.event_index,
            ), dtype=EventInfo.dtype))

        self.event_index += 1
        blob['EventInfo'] = event_info
        blob['Hits'] = hit_series
        return blob

    def extract_timeslice_frame(self):
        blob = Blob()
        r = self.timeslice_reader
        n = r.number_of_hits
        channel_ids = np.zeros(n, dtype='i')
        dom_ids = np.zeros(n, dtype='i')
        times = np.zeros(n, dtype='i')
        tots = np.zeros(n, dtype='i')
        r.get_hits(channel_ids, dom_ids, times, tots)
        hit_series = TimesliceHitSeries.from_arrays(
            channel_ids, dom_ids, times, tots,
            self.timeslice_index, self.timeslice_frame_index
        )
        timesliceframe_info = TimesliceFrameInfo(
                r.dom_id,
                r.fifo_status,
                self.timeslice_frame_index,
                r.frame_index,
                r.has_udp_trailer,
                r.high_rate_veto,
                r.max_sequence_number,
                r.number_of_received_packets,
                self.timeslice_index,
                r.utc_nanoseconds,
                r.utc_seconds,
                r.white_rabbit_status,
                )

        blob['TimesliceHits'] = hit_series
        blob['TimesliceFrameInfo'] = timesliceframe_info
        return blob

    def extract_summaryslice_frame(self):
        blob = Blob()
        r = self.summaryslice_reader
        summaryframe_info = SummaryframeInfo(
                r.dom_id,
                r.fifo_status,
                self.summaryslice_frame_index,
                r.frame_index,
                r.has_udp_trailer,
                r.high_rate_veto,
                r.max_sequence_number,
                r.number_of_received_packets,
                self.summaryslice_index,
                r.utc_nanoseconds,
                r.utc_seconds,
                r.white_rabbit_status,
                )

        blob['SummaryframeInfo'] = summaryframe_info
        return blob

    def process(self, blob):
        return next(self.blobs)

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        return next(self.blobs)


class EventPump(Pump):
    """A pump for DAQEvents in JPP files.

    Parameters
    ----------
    filename: str
        Name of the file to open.

    """
    def configure(self):

        try:
            import jppy  # noqa
        except ImportError:
            raise ImportError("\nEither Jpp or jppy could not be found."
                              "\nMake sure you source the JPP environmanet "
                              "and have jppy installed")

        self.event_index = self.get('index') or 0
        self.filename = self.require('filename')

        self.event_reader = jppy.PyJDAQEventReader(self.filename)
        self.blobs = self.blob_generator()

    def blob_generator(self):
        while self.event_reader.has_next:
            try:
                yield self.extract_event()
            except IndexError:
                pass

        raise StopIteration

    def extract_event(self):
        blob = Blob()
        r = self.event_reader
        r.retrieve_next_event()  # do it at the beginning!

        n = r.number_of_snapshot_hits
        channel_ids = np.zeros(n, dtype='i')
        dom_ids = np.zeros(n, dtype='i')
        times = np.zeros(n, dtype='i')
        tots = np.zeros(n, dtype='i')
        triggereds = np.zeros(n, dtype='i')

        r.get_hits(channel_ids, dom_ids, times, tots, triggereds)

        hit_series = RawHitSeries.from_arrays(
            channel_ids, dom_ids, times, tots, triggereds, self.event_index
        )

        event_info = EventInfo(np.array((
            r.det_id, r.frame_index,
            0,  # livetime_sec
            0, 0,  # MC ID and time
            0,  # n_events_gen
            0,  # n_files_gen
            r.overlays,
            r.trigger_counter, r.trigger_mask,
            r.utc_nanoseconds, r.utc_seconds,
            np.nan, np.nan, np.nan,   # w1-w3
            0,  # run
            self.event_index,
            ), dtype=EventInfo.dtype))

        self.event_index += 1
        blob['EventInfo'] = event_info
        blob['Hits'] = hit_series
        return blob

    def process(self, blob):
        return next(self.blobs)

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        return next(self.blobs)


class TimeslicePump(Pump):
    """A pump to read and extract timeslices. Currently only hits are read.

    Required Parameters
    -------------------
    filename: str

    """
    def configure(self):
        filename = self.require('filename')
        self.blobs = self.timeslice_generator()
        try:
            import jppy  # noqa
        except ImportError:
            raise ImportError("\nEither Jpp or jppy could not be found."
                              "\nMake sure you source the JPP environmanet "
                              "and have jppy installed")
        self.r = jppy.daqtimeslicereader.PyJDAQTimesliceReader(filename)

    def process(self, blob):
        return next(self.blobs)

    def timeslice_generator(self):
        buf_size = 5000
        channel_ids = np.zeros(buf_size, dtype='i')
        dom_ids = np.zeros(buf_size, dtype='i')
        times = np.zeros(buf_size, dtype='i')
        tots = np.zeros(buf_size, dtype='i')
        triggereds = np.zeros(buf_size, dtype=bool)
        while self.r.has_next:
            slice_id = 1
            blob = Blob()
            self.r.retrieve_next_timeslice()
            n_frames = 0
            total_hits = 0
            while self.r.has_next_superframe:
                n_frames += 1
                n = self.r.number_of_hits
                if n != 0:
                    start_index = total_hits
                    total_hits += n
                    if total_hits > buf_size:
                        channel_ids.resize(total_hits)
                        dom_ids.resize(total_hits)
                        times.resize(total_hits)
                        tots.resize(total_hits)
                        triggereds.resize(total_hits)
                    self.r.get_hits(channel_ids, dom_ids, times, tots,
                                    start_index)
                self.r.retrieve_next_superframe()

            hits = RawHitSeries.from_arrays(channel_ids[:total_hits],
                                            dom_ids[:total_hits],
                                            times[:total_hits],
                                            tots[:total_hits],
                                            triggereds[:total_hits],
                                            slice_id)
            blob['TSHits'] = hits
            yield blob
            slice_id += 1

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        return next(self.blobs)


class SummaryslicePump(Pump):
    """Preliminary Summaryslice reader"""
    def configure(self):
        filename = self.require('filename')
        self.blobs = self.summaryslice_generator()
        try:
            import jppy  # noqa
        except ImportError:
            raise ImportError("\nEither Jpp or jppy could not be found."
                              "\nMake sure you source the JPP environmanet "
                              "and have jppy installed")
        self.r = jppy.daqsummaryslicereader.PyJDAQSummarysliceReader(filename)

    def process(self, blob):
        return next(self.blobs)

    def summaryslice_generator(self):
        while self.r.has_next:
            summary_slice = {}
            self.r.retrieve_next_summaryslice()
            blob = Blob()
            while self.r.has_next_frame:
                rates = np.zeros(31, dtype='f8')
                self.r.get_rates(rates)
                summary_slice[self.r.dom_id] = {
                        'rates': rates,
                        'n_udp_packets': self.r.number_of_received_packets,
                        'max_sequence_number': self.r.max_sequence_number,
                        'has_udp_trailer': self.r.has_udp_trailer,
                        'high_rate_veto': self.r.high_rate_veto,
                        'fifo_status': self.r.fifo_status,
                        'frame_index': self.r.frame_index,
                        'utc_seconds': self.r.utc_seconds,
                        'utc_nanoseconds': self.r.utc_nanoseconds,
                        }
                self.r.retrieve_next_frame()
            blob['Summaryslice'] = summary_slice
            yield blob

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        return next(self.blobs)
