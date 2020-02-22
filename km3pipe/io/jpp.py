#!/usr/bin/env python
# Filename: jpp.py
# pylint: disable=
"""
Pump for the jpp file read through aanet interface.

"""
from __future__ import absolute_import, print_function, division

import numpy as np

from km3pipe.core import Pump, Blob
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

        self.event_reader = km3io.OfflineReader(self.filename.encode())
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

    def extract_event(self, event_number):
        blob = self._current_blob
        r = self.event_reader
        hits = r.hits[event_number]
        event = r.events[event_number]

        hit_series = Table.from_template({
            'channel_id': hits.channel_id,
            'dom_id': hits.dom_id,
            'time': hits.tdc,
            'tot': hits.tot,
            'triggered': hits.trig,
            'group_id': self.event_index,
        }, 'Hits')

        event_info = Table.from_template({
            'det_id': event.det_id,
            'frame_index': event.frame_index,
            'livetime_sec': 0,
            'mc_id': 0,
            'mc_t': 0,
            'n_events_gen': 0,
            'n_files_gen': 0,
            'overlays': event.overlays,
            'trigger_counter': event.trigger_counter,
            'trigger_mask': event.trigger_mask,
            'utc_nanoseconds': events.t_fNanoSec,
            'utc_seconds': events.t_fSec,
            'weight_w1': np.nan,
            'weight_w2': np.nan,
            'weight_w3': np.nan,
            'run_id': event.run_id,
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

    def create_timeslice_info(self):
        header = self.r.timeslices.stream(self.stream, 0).header
        frame_ids = header['frame_index'].array()
        number_of_frames = len(frame_ids)
        timestamps = header['timeslice_start.UTC_seconds'].array()
        nanoseconds = header['timeslice_start.UTC_16nanosecondcycles'].array()
        timeslice_info = Table.from_template({
            'frame_index': frame_ids,
            'slice_id': range(number_of_frames),
            'timestamp': timestamps,
            'nanoseconds': nanoseconds,
            'n_frames': number_of_frames * np.ones(number_of_frames),
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
        blob[self._hits_blob_key] = hits
        return blob

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


class FitPump(Pump):
    """A pump for JFit objects in JPP files.

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

        self.buf_size = 50000
        self._pos_xs = np.zeros(self.buf_size, dtype='d')
        self._pos_ys = np.zeros(self.buf_size, dtype='d')
        self._pos_zs = np.zeros(self.buf_size, dtype='d')
        self._dir_xs = np.zeros(self.buf_size, dtype='d')
        self._dir_ys = np.zeros(self.buf_size, dtype='d')
        self._dir_zs = np.zeros(self.buf_size, dtype='d')
        self._ndfs = np.zeros(self.buf_size, dtype='i')
        self._times = np.zeros(self.buf_size, dtype='d')
        self._qualities = np.zeros(self.buf_size, dtype='d')
        self._energies = np.zeros(self.buf_size, dtype='d')

        self.event_reader = jppy.PyJFitReader(self.filename.encode())
        self.blobs = self.blob_generator()

    def _resize_buffers(self, buf_size):
        log.info("Resizing hit buffers to {}.".format(buf_size))
        self.buf_size = buf_size
        self._pos_xs.resize(buf_size, refcheck=False)
        self._pos_ys.resize(buf_size, refcheck=False)
        self._pos_zs.resize(buf_size, refcheck=False)
        self._dir_xs.resize(buf_size, refcheck=False)
        self._dir_ys.resize(buf_size, refcheck=False)
        self._dir_zs.resize(buf_size, refcheck=False)
        self._ndfs.resize(buf_size, refcheck=False)
        self._times.resize(buf_size, refcheck=False)
        self._qualities.resize(buf_size, refcheck=False)
        self._energies.resize(buf_size, refcheck=False)

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
        r.retrieve_next_event()    # do it at the beginning!

        n = r.n_fits

        if n > self.buf_size:
            self._resize_buffers(int(n * 3 / 2))

        r.get_fits(
            self._pos_xs,
            self._pos_ys,
            self._pos_zs,
            self._dir_xs,
            self._dir_ys,
            self._dir_zs,
            self._ndfs,
            self._times,
            self._qualities,
            self._energies,
        )
        fit_collection = Table({
            'pos_x': self._pos_xs[:n],
            'pos_y': self._pos_ys[:n],
            'pos_z': self._pos_zs[:n],
            'dir_x': self._dir_xs[:n],
            'dir_y': self._dir_ys[:n],
            'dir_z': self._dir_zs[:n],
            'ndf': self._ndfs[:n],
            'time': self._times[:n],
            'quality': self._qualities[:n],
            'energy': self._energies[:n],
        },
                               h5loc='/jfit')
        fit_collection = fit_collection.append_columns(['event_id'],
                                                       [self.event_index])

        # TODO make this into a datastructure

        event_info = Table.from_template({
            'det_id': 0,
            'frame_index': 0,
            'livetime_sec': 0,
            'MC ID': 0,
            'MC time': 0,
            'n_events_gen': 0,
            'n_files_gen': 0,
            'overlays': 0,
            'trigger_counter': 0,
            'trigger_mask': 0,
            'utc_nanoseconds': 0,
            'utc_seconds': 0,
            'weight_w1': np.nan,
            'weight_w2': np.nan,
            'weight_w3': np.nan,
            'run_id': 0,
            'group_id': self.event_index,
        }, 'EventInfo')

        self.event_index += 1
        blob['EventInfo'] = event_info
        blob['JFit'] = fit_collection
        return blob

    def process(self, blob):
        nextblob = next(self.blobs)
        blob.update(nextblob)
        return blob

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.blobs)
