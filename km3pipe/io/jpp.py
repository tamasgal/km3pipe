#!/usr/bin/env python
# Filename: jpp.py
# pylint: disable=
"""
Pump for the jpp file read through aanet interface.

"""

import numpy as np

from km3pipe.core import Pump, Blob
from km3pipe.dataclasses import Table
from km3pipe.logger import get_logger

log = get_logger(__name__)    # pylint: disable=C0103

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = [
    "Thomas Heid", "Giuliano Maggi", "Moritz Lotze", "Johannes Schumann"
]
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
            import km3io
        except ImportError:
            raise ImportError(
                "\nKM3iopackage could not be found."
                "\n Make sure the km3io package is installed"
            )

        self.event_index = self.get('index') or 0
        self.filename = self.require('filename')

        self.event_reader = km3io.DAQReader(self.filename.encode())
        self.blobs = self.blob_generator()
        self.n_events = len(self.event_reader.events)
        self._current_blob = Blob()

    def blob_generator(self):
        for i in range(self.n_events):
            try:
                yield self.extract_event(i)
            except IndexError:
                pass

        raise StopIteration

    def __getitem__(self, index):
        if index >= self.n_events:
            raise IndexError
        return self.extract_event(index)

    def _get_trigger_mask(self, snapshot_hits, triggered_hits):
        trg_mask = np.zeros(len(snapshot_hits))
        s = np.array([
            snapshot_hits.time, snapshot_hits.channel_id, snapshot_hits.dom_id
        ]).transpose()
        t = np.array([
            triggered_hits.time, triggered_hits.channel_id,
            triggered_hits.dom_id
        ]).transpose()
        cmp_mask = (s == t[:, None])
        trg_map = np.argwhere(np.all(cmp_mask, axis=2))[:, 1]
        trg_mask[trg_map] = triggered_hits.trigger_mask
        return trg_mask

    def extract_event(self, event_number):
        blob = self._current_blob
        r = self.event_reader
        hits = r.events.snapshot_hits[event_number]
        trg_hits = r.events.triggered_hits[event_number]
        raw_event_info = r.events.headers[event_number]

        trigger_mask = self._get_trigger_mask(hits, trg_hits)
        hit_series = Table({
            'channel_id': hits.channel_id,
            'dom_id': hits.dom_id,
            'time': hits.time,
            'tot': hits.tot,
            'triggered': trigger_mask,
            'group_id': self.event_index,
        })

        event_info = Table.from_template({
            'det_id': raw_event_info["detector_id"],
            'frame_index': raw_event_info["frame_index"],
            'livetime_sec': 0,
            'mc_id': 0,
            'mc_t': 0,
            'n_events_gen': 0,
            'n_files_gen': 0,
            'overlays': raw_event_info["overlays"],
            'trigger_counter': raw_event_info["trigger_counter"],
            'trigger_mask': raw_event_info["trigger_mask"],
            'utc_nanoseconds': raw_event_info["UTC_16nanosecondcycles"] * 16.0,
            'utc_seconds': raw_event_info["UTC_seconds"],
            'weight_w1': np.nan,
            'weight_w2': np.nan,
            'weight_w3': np.nan,
            'run_id': raw_event_info["run"],
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


class TimeslicePump(Pump):
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
            import km3io
        except ImportError:
            raise ImportError(
                "\nKM3iopackage could not be found."
                "\n Make sure the km3io package is installed"
            )
        self.r = km3io.DAQReader(fname)
        self.timeslice_info = self.create_timeslice_info()
        self.n_timeslices = len(self.timeslice_info)

        self._current_blob = Blob()
        self._hits_blob_key = '{}Hits'.format(
            self.stream if self.stream else 'TS'
        )

    def create_timeslice_info(self):
        header = self.r.timeslices.stream(self.stream, 0).header
        slice_ids = header['frame_index'].array()
        timestamps = header['timeslice_start.UTC_seconds'].array()
        number_of_slices = len(slice_ids)
        nanoseconds = header['timeslice_start.UTC_16nanosecondcycles'].array(
        ) * 16
        timeslice_info = Table.from_template({
            'frame_index': slice_ids,
            'slice_id': range(number_of_slices),
            'timestamp': timestamps,
            'nanoseconds': nanoseconds,
            'n_frames': np.zeros(len(slice_ids)),
        }, 'TimesliceInfo')
        return timeslice_info

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
        hits = self._extract_hits(index)
        hits.group_id = index
        blob['TimesliceInfo'] = self.timeslice_info[index:index + 1]
        blob['TimesliceInfo']['n_frames'] = self._extract_number_of_frames(
            index
        )
        blob[self._hits_blob_key] = hits
        return blob

    def _extract_number_of_frames(self, index):
        timeslice = self.r.timeslices.stream(self.stream, index)
        return len(timeslice.frames)

    def _extract_hits(self, index):
        timeslice = self.r.timeslices.stream(self.stream, index)
        raw_hits = {
            "dom_id": [],
            "channel_id": [],
            "time": [],
            "tot": [],
            "group_id": []
        }

        for dom_id, frame in timeslice.frames.items():
            raw_hits['channel_id'].extend(frame.pmt)
            raw_hits['time'].extend(frame.tdc)
            raw_hits['tot'].extend(frame.tot)
            raw_hits['dom_id'].extend(len(frame.pmt) * [dom_id])
            raw_hits['group_id'].extend(len(frame.pmt) * [0])

        hits = Table.from_template(raw_hits, 'TimesliceHits')
        return hits

    def get_by_frame_index(self, frame_index):
        blob = Blob()
        ts_info = self.timeslice_info[self.timeslice_info.frame_index ==
                                      frame_index][0]
        slice_id = ts_info.slice_id
        hits = self._extract_hits(slice_id)
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


class SummaryslicePump(Pump):
    """Preliminary Summaryslice reader"""
    def configure(self):
        filename = self.require('filename')
        self.blobs = self.summaryslice_generator()
        try:
            import km3io
        except ImportError:
            raise ImportError(
                "\nKM3iopackage could not be found."
                "\n Make sure the km3io package is installed"
            )
        self.r = km3io.DAQReader(filename)
        self.n_summaryslices = len(self.r.summaryslices.slices)
        self.summaryslice_info = self._create_summaryslice_info()

    def process(self, blob):
        return next(self.blobs)

    def _create_summaryslice_info(self):
        header = self.r.summaryslices.headers
        frame_ids = np.array(header["frame_index"])
        timestamps = np.array(header["UTC_seconds"])
        nanoseconds = np.array(header["UTC_16nanosecondcycles"])
        summaryslice_info = Table.from_template({
            'frame_index': frame_ids,
            'slice_id': range(self.n_summaryslices),
            'timestamp': timestamps,
            'nanoseconds': nanoseconds,
            'n_frames': [len(v) for v in self.r.summaryslices.slices.dom_id],
        }, 'SummarysliceInfo')
        return summaryslice_info

    def summaryslice_generator(self):
        try:
            import km3io
        except ImportError:
            raise ImportError(
                "\nKM3iopackage could not be found."
                "\n Make sure the km3io package is installed"
            )
        for i in range(self.n_summaryslices):
            blob = Blob()
            blob['SummarysliceInfo'] = self.summaryslice_info[i:i + 1]
            raw_summaryslice = self.r.summaryslices.slices[i]
            summary_slice = {}
            for dom_id in raw_summaryslice.dom_id:
                frame = raw_summaryslice[raw_summaryslice.dom_id == dom_id]
                raw_rates = [getattr(frame, 'ch%d' % i)[0] for i in range(31)]
                rates = km3io.daq.get_rate(raw_rates).astype(np.float64)
                hrvs = km3io.daq.get_channel_flags(frame.hrv)
                fifos = km3io.daq.get_channel_flags(frame.fifo)
                udp_packets = km3io.daq.get_number_udp_packets(frame.dq_status)
                max_sequence_number = km3io.daq.get_udp_max_sequence_number(
                    frame.dq_status
                )
                has_udp_trailer = km3io.daq.has_udp_trailer(frame.fifo)
                summary_slice[dom_id] = {
                    'rates': rates,
                    'hrvs': hrvs,
                    'fifos': fifos,
                    'n_udp_packets': udp_packets,
                    'max_sequence_number': max_sequence_number,
                    'has_udp_trailer': has_udp_trailer,
                    'high_rate_veto': np.any(hrvs),
                    'fifo_status': np.any(fifos),
                }
            blob['Summaryslice'] = summary_slice
            yield blob

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.blobs)
