#!/usr/bin/env python
# Filename: jpp.py
# pylint: disable=
"""
Pump for the jpp file read through aanet interface.

"""
from __future__ import absolute_import, print_function, division

import numpy as np

from km3pipe.core import Blob, Module
from km3pipe.dataclasses import Table
from km3pipe.logger import get_logger

log = get_logger(__name__)    # pylint: disable=C0103

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Thomas Heid", "Giuliano Maggi", "Moritz Lotze"]
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class EventPump(Module):
    """A pump for DAQEvents in JPP files.

    Parameters
    ----------
    filename: str
        Name of the file to open.

    """
    def configure(self):

        try:
            import jppy    # noqa
        except ImportError:
            raise ImportError(
                "\nEither Jpp or jppy could not be found."
                "\nMake sure you source the JPP environmanet "
                "and have jppy installed"
            )

        self.event_index = self.get('index') or 0
        self.filename = self.require('filename')

        self.buf_size = 5000
        self._channel_ids = np.zeros(self.buf_size, dtype='i')
        self._dom_ids = np.zeros(self.buf_size, dtype='i')
        self._times = np.zeros(self.buf_size, dtype='i')
        self._tots = np.zeros(self.buf_size, dtype='i')
        self._triggereds = np.zeros(self.buf_size, dtype='i')

        self.event_reader = jppy.PyJDAQEventReader(self.filename.encode())
        self.blobs = self.blob_generator()
        self._current_blob = Blob()

    def _resize_buffers(self, buf_size):
        log.info("Resizing hit buffers to {}.".format(buf_size))
        self.buf_size = buf_size
        self._channel_ids.resize(buf_size, refcheck=False)
        self._dom_ids.resize(buf_size, refcheck=False)
        self._times.resize(buf_size, refcheck=False)
        self._tots.resize(buf_size, refcheck=False)
        self._triggereds.resize(buf_size, refcheck=False)

    def blob_generator(self):
        while self.event_reader.has_next:
            try:
                yield self.extract_event()
            except IndexError:
                pass

        raise StopIteration

    def extract_event(self):
        blob = self._current_blob
        r = self.event_reader
        r.retrieve_next_event()    # do it at the beginning!

        n = r.number_of_snapshot_hits

        if n > self.buf_size:
            self._resize_buffers(int(n * 3 / 2))

        r.get_hits(
            self._channel_ids, self._dom_ids, self._times, self._tots,
            self._triggereds
        )

        hit_series = Table.from_template({
            'channel_id': self._channel_ids[:n],
            'dom_id': self._dom_ids[:n],
            'time': self._times[:n],
            'tot': self._tots[:n],
            'triggered': self._triggereds[:n],
            'group_id': self.event_index,
        }, 'Hits')

        event_info = Table.from_template({
            'det_id': r.det_id,
            'frame_index': r.frame_index,
            'livetime_sec': 0,
            'mc_id': 0,
            'mc_t': 0,
            'n_events_gen': 0,
            'n_files_gen': 0,
            'overlays': r.overlays,
            'trigger_counter': r.trigger_counter,
            'trigger_mask': r.trigger_mask,
            'utc_nanoseconds': r.utc_nanoseconds,
            'utc_seconds': r.utc_seconds,
            'weight_w1': np.nan,
            'weight_w2': np.nan,
            'weight_w3': np.nan,
            'run_id': 0,
            'group_id': self.event_index,
        }, 'EventInfo')

        self.event_index += 1
        blob['EventInfo'] = event_info
        blob['Hits'] = hit_series
        return blob

    def process(self, blob):
        self._current_blob = blob
        return next(self.blobs)

    def __iter__(self):
        return self

    def __next__(self):
        self._current_blob = next(self.blobs)
        return self._current_blob


class TimeslicePump(Module):
    """A pump to read and extract timeslices. Currently only hits are read.

    Required Parameters
    -------------------
    filename: str
    stream: str ('L0', 'L1', 'L2', 'SN') default: 'L1'

    """
    def configure(self):
        fname = self.require('filename')
        self.stream = self.get('stream', default='L1')
        self.blobs = self.timeslice_generator()
        try:
            import jppy    # noqa
        except ImportError:
            raise ImportError(
                "\nEither Jpp or jppy could not be found."
                "\nMake sure you source the JPP environmanet "
                "and have jppy installed"
            )
        stream = 'JDAQTimeslice' + self.stream
        self.r = jppy.daqtimeslicereader.PyJDAQTimesliceReader(
            fname.encode(), stream.encode()
        )
        self.n_timeslices = self.r.n_timeslices

        self.buf_size = 5000
        self._channel_ids = np.zeros(self.buf_size, dtype='i')
        self._dom_ids = np.zeros(self.buf_size, dtype='i')
        self._times = np.zeros(self.buf_size, dtype='i')
        self._tots = np.zeros(self.buf_size, dtype='i')
        self._triggereds = np.zeros(self.buf_size, dtype=bool)    # dummy

        self._current_blob = Blob()
        self._hits_blob_key = '{}Hits'.format(
            self.stream if self.stream else 'TS'
        )

    def process(self, blob):
        self._current_blob = blob
        return next(self.blobs)

    def timeslice_generator(self):
        """Uses slice ID as iterator"""
        slice_id = 0
        while slice_id < self.n_timeslices:
            blob = self.get_blob(slice_id)
            yield blob
            slice_id += 1

    def get_blob(self, index):
        """Index is slice ID"""
        blob = self._current_blob
        self.r.retrieve_timeslice(index)
        timeslice_info = Table.from_template({
            'frame_index': self.r.frame_index,
            'slice_id': index,
            'timestamp': self.r.utc_seconds,
            'nanoseconds': self.r.utc_nanoseconds,
            'n_frames': self.r.n_frames,
        }, 'TimesliceInfo')
        hits = self._extract_hits()
        hits.group_id = index
        blob['TimesliceInfo'] = timeslice_info
        blob[self._hits_blob_key] = hits
        return blob

    def _extract_hits(self):
        total_hits = self.r.number_of_hits

        if total_hits > self.buf_size:
            buf_size = int(total_hits * 3 / 2)
            self._resize_buffers(buf_size)

        self.r.get_hits(
            self._channel_ids, self._dom_ids, self._times, self._tots
        )

        group_id = 0 if total_hits > 0 else []

        hits = Table.from_template(
            {
                'channel_id': self._channel_ids[:total_hits],
                'dom_id': self._dom_ids[:total_hits],
                'time': self._times[:total_hits].astype('f8'),
                'tot': self._tots[:total_hits],
        # 'triggered': self._triggereds[:total_hits],  # dummy
                'group_id': group_id,    # slice_id will be set afterwards
            },
            'TimesliceHits'
        )
        return hits

    def _resize_buffers(self, buf_size):
        log.info("Resizing hit buffers to {}.".format(buf_size))
        self.buf_size = buf_size
        self._channel_ids.resize(buf_size, refcheck=False)
        self._dom_ids.resize(buf_size, refcheck=False)
        self._times.resize(buf_size, refcheck=False)
        self._tots.resize(buf_size, refcheck=False)
        self._triggereds.resize(buf_size, refcheck=False)    # dummy

    def get_by_frame_index(self, frame_index):
        blob = Blob()
        self.r.retrieve_timeslice_at_frame_index(frame_index)
        hits = self._extract_hits()
        blob[self._hits_blob_key] = hits
        return blob

    def __len__(self):
        return self.n_timeslices

    def __iter__(self):
        return self

    def __next__(self):
        self._current_blob = next(self.blobs)
        return self._current_blob

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


class SummaryslicePump(Module):
    """Preliminary Summaryslice reader"""
    def configure(self):
        filename = self.require('filename')
        self.blobs = self.summaryslice_generator()
        try:
            from jppy.daqsummaryslicereader import PyJDAQSummarysliceReader
        except ImportError:
            raise ImportError(
                "\nEither Jpp or jppy could not be found."
                "\nMake sure you source the JPP environmanet "
                "and have jppy installed"
            )
        self.r = PyJDAQSummarysliceReader(filename.encode())

    def process(self, blob):
        return next(self.blobs)

    def summaryslice_generator(self):
        slice_id = 0
        while self.r.has_next:
            summary_slice = {}
            self.r.retrieve_next_summaryslice()
            blob = Blob()
            summaryslice_info = Table.from_template({
                'frame_index': self.r.frame_index,
                'slice_id': slice_id,
                'timestamp': self.r.utc_seconds,
                'nanoseconds': self.r.utc_nanoseconds,
                'n_frames': self.r.n_frames,
            }, 'SummarysliceInfo')
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

    def __next__(self):
        return next(self.blobs)
