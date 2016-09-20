#!/usr/bin/env python
# coding=utf-8
# Filename: jpp.py
# pylint: disable=
"""
Pump for the jpp file read through aanet interface.

"""
from __future__ import division, absolute_import, print_function

import numpy as np

from km3pipe import Pump, Blob
from km3pipe.dataclasses import EventInfo, HitSeries, L0HitSeries
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
    """A pump for JPP ROOT files."""

    def __init__(self, filename, **context):
        super(self.__class__, self).__init__(**context)

        try:
            import jppy  # noqa
        except ImportError:
            raise ImportError("\nPlease install the jppy package:\n\n"
                              "    pip install jppy\n")

        self.event_index = self.get('index') or 0
        self.timeslice_index = 0
        # self.filename = self.get('filename')
        self.filename = filename

        self.event_reader = jppy.PyJDAQEventReader(self.filename)
        self.timeslice_reader = jppy.PyJDAQTimesliceReader(self.filename)
        self.blobs = self.blob_generator()

    def blob_generator(self):
        while self.event_reader.has_next:
            yield self.extract_event()
        while self.timeslice_reader.has_next:
            self.timeslice_reader.retrieve_next_timeslice()
            while self.timeslice_reader.has_next_superframe:
                yield self.extract_frame()
                self.timeslice_reader.retrieve_next_superframe()
            self.timeslice_index += 1
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

        hit_series = HitSeries.from_arrays(
            channel_ids, dom_ids, np.arange(n), np.zeros(n), times,
            tots, triggereds, self.event_index
        )

        event_info = EventInfo(
            r.det_id, r.frame_index,
            0, 0,  # MC ID and time
            r.overlays,
            # r.run_id,
            r.trigger_counter, r.trigger_mask,
            r.utc_nanoseconds, r.utc_seconds,
            np.nan, np.nan, np.nan,   # w1-w3
            self.event_index,
            )

        self.event_index += 1
        blob['EventInfo'] = event_info
        blob['Hits'] = hit_series
        return blob

    def extract_frame(self):
        blob = Blob()
        r = self.timeslice_reader
        n = r.number_of_hits
        channel_ids = np.zeros(n, dtype='i')
        dom_ids = np.zeros(n, dtype='i')
        times = np.zeros(n, dtype='i')
        tots = np.zeros(n, dtype='i')
        triggereds = np.zeros(n, dtype='i')
        r.get_hits(channel_ids, dom_ids, times, tots)
        hit_series = L0HitSeries.from_arrays(
            channel_ids, dom_ids, times, tots, self.timeslice_index
        )

        blob['L0Hits'] = hit_series
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
