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
from km3pipe.dataclasses import (EventInfo, TimesliceInfo, SummarysliceInfo,
                                 RawHitSeries)
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Thomas Heid"]
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


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

        self.buf_size = 5000
        self._channel_ids = np.zeros(self.buf_size, dtype='i')
        self._dom_ids = np.zeros(self.buf_size, dtype='i')
        self._times = np.zeros(self.buf_size, dtype='i')
        self._tots = np.zeros(self.buf_size, dtype='i')
        self._triggereds = np.zeros(self.buf_size, dtype='i')

        self.event_reader = jppy.PyJDAQEventReader(self.filename)
        self.blobs = self.blob_generator()

    def _resize_buffers(self, buf_size):
        log.info("Resizing hit buffers to {}.".format(buf_size))
        self.buf_size = buf_size
        self._channel_ids.resize(buf_size)
        self._dom_ids.resize(buf_size)
        self._times.resize(buf_size)
        self._tots.resize(buf_size)
        self._triggereds.resize(buf_size)

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

        if n > self.buf_size:
            self._resize_buffers(int(n * 3 / 2))

        r.get_hits(self._channel_ids,
                   self._dom_ids,
                   self._times,
                   self._tots,
                   self._triggereds)

        hit_series = RawHitSeries.from_arrays(
            self._channel_ids[:n],
            self._dom_ids[:n],
            self._times[:n],
            self._tots[:n],
            self._triggereds[:n],
            self.event_index
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
        self._scanner_initialised = False

        self.buf_size = 5000
        self._channel_ids = np.zeros(self.buf_size, dtype='i')
        self._dom_ids = np.zeros(self.buf_size, dtype='i')
        self._times = np.zeros(self.buf_size, dtype='i')
        self._tots = np.zeros(self.buf_size, dtype='i')
        self._triggereds = np.zeros(self.buf_size, dtype=bool)

    def process(self, blob):
        return next(self.blobs)

    def timeslice_generator(self):
        slice_id = 0
        while self.r.has_next:
            blob = Blob()
            self.r.retrieve_next_timeslice()
            timeslice_info = TimesliceInfo(
                    frame_index=self.r.frame_index,
                    slice_id=slice_id,
                    timestamp=self.r.utc_seconds,
                    nanoseconds=self.r.utc_nanoseconds,
                    n_frames=self.r.n_frames,
                    )
            hits = self._extract_hits()
            hits.slice_id = slice_id
            blob['TimesliceInfo'] = timeslice_info
            blob['TSHits'] = hits
            yield blob
            slice_id += 1

    def _extract_hits(self):
        n_frames = 0
        total_hits = 0
        while self.r.has_next_superframe:
            n_frames += 1
            n = self.r.number_of_hits
            if n != 0:
                start_index = total_hits
                total_hits += n
                if total_hits > self.buf_size:
                    buf_size = int(total_hits * 3 / 2)
                    self._resize_buffers(buf_size)
                self.r.get_hits(self._channel_ids,
                                self._dom_ids,
                                self._times,
                                self._tots,
                                start_index)
            self.r.retrieve_next_superframe()

        hits = RawHitSeries.from_arrays(self._channel_ids[:total_hits],
                                        self._dom_ids[:total_hits],
                                        self._times[:total_hits],
                                        self._tots[:total_hits],
                                        self._triggereds[:total_hits],
                                        0)
        return hits

    def _resize_buffers(self, buf_size):
        log.info("Resizing hit buffers to {}.".format(buf_size))
        self.buf_size = buf_size
        self._channel_ids.resize(buf_size)
        self._dom_ids.resize(buf_size)
        self._times.resize(buf_size)
        self._tots.resize(buf_size)
        self._triggereds.resize(buf_size)

    def get_by_frame_index(self, frame_index):
        if not self._scanner_initialised:
            self.r.init_tree_scanner()
            self._scanner_initialised = True
        blob = Blob()
        self.r.retrieve_timeslice_at_frame_index(frame_index)
        hits = self._extract_hits()
        blob['TSHits'] = hits
        return blob

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
        slice_id = 0
        while self.r.has_next:
            summary_slice = {}
            self.r.retrieve_next_summaryslice()
            blob = Blob()
            summaryslice_info = SummarysliceInfo(
                    frame_index=self.r.frame_index,
                    slice_id=slice_id,
                    timestamp=self.r.utc_seconds,
                    nanoseconds=self.r.utc_nanoseconds,
                    n_frames=self.r.n_frames,
                    )
            blob['SummarysliceInfo'] = summaryslice_info
            while self.r.has_next_frame:
                rates = np.zeros(31, dtype='f8')
                hrvs = np.zeros(31, dtype='i4')
                fifos = np.zeros(31, dtype='i4')
                self.r.get_rates(rates)
                self.r.get_hrvs(hrvs)
                self.r.get_fifos(fifos)
                summary_slice[self.r.dom_id] = {
                        'rates': rates,
                        'hrvs': hrvs.astype(bool),
                        'fifos': fifos.astype(bool),
                        'n_udp_packets': self.r.number_of_received_packets,
                        'max_sequence_number': self.r.max_sequence_number,
                        'has_udp_trailer': self.r.has_udp_trailer,
                        'high_rate_veto': self.r.high_rate_veto,
                        'fifo_status': self.r.fifo_status,
                        }
                self.r.retrieve_next_frame()
            blob['Summaryslice'] = summary_slice
            slice_id += 1
            yield blob

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        return next(self.blobs)
