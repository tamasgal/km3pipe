#!/usr/bin/env python
# coding=utf-8
# Filename: jpp.py
# pylint: disable=
"""
Pump for the jpp file read through aanet interface.

"""
from __future__ import division, absolute_import, print_function

import numpy as np

from km3pipe import Pump
from km3pipe.dataclasses import EventInfo, HitSeries
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

        self.index = self.get('index') or 0
        # self.filename = self.get('filename')
        self.filename = filename

        self.reader = jppy.PyJDAQEventReader(self.filename)
        self.blobs = self.blob_generator()

    def blob_generator(self):
        while self.reader.has_next:
            r = self.reader
            r.retrieve_next_event()

            n = r.number_of_snapshot_hits
            channel_ids = np.zeros(n, dtype='i')
            dom_ids = np.zeros(n, dtype='i')
            times = np.zeros(n, dtype='i')
            tots = np.zeros(n, dtype='i')
            triggereds = np.zeros(n, dtype='i')

            r.get_hits(channel_ids, dom_ids, times, tots, triggereds)

            hit_series = HitSeries.from_arrays(
                channel_ids, dom_ids, np.arange(n), np.zeros(n), times, tots,
                triggereds, self.index
            )

            event_info = EventInfo(r.det_id, self.index, r.frame_index,
                                   0, 0,  # MC ID and time
                                   r.overlays, r.run_id,
                                   r.trigger_counter, r.trigger_mask,
                                   r.utc_nanoseconds, r.utc_seconds,
                                   np.nan, np.nan, np.nan   # w1-w3
                                   )

            self.index += 1
            yield {'EventInfo': event_info, 'Hits': hit_series}

    def process(self, blob):
        return next(self.blobs)

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        return next(self.blobs)
