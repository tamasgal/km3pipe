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
from km3pipe.dataclasses import HitSeries
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103


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
        #self.filename = self.get('filename')
        self.filename = filename

        self.reader = jppy.PyJDAQEventReader(self.filename)
        self.blobs = self.blob_generator()

    def blob_generator(self):
        while self.reader.has_next():
            self.reader.retrieve_next_event()
            self.index += 1

            n = self.reader.get_number_of_snapshot_hits()
            channel_ids = np.zeros(n, dtype='i')
            dom_ids = np.zeros(n, dtype='i')
            times = np.zeros(n, dtype='i')
            tots = np.zeros(n, dtype='i')
            triggereds = np.zeros(n, dtype='i')

            self.reader.get_hits(channel_ids, dom_ids, times, tots, triggereds)

            hit_series = HitSeries.from_arrays(
                channel_ids, dom_ids, np.arange(n), np.zeros(n), times, tots,
                triggereds, self.index
            )

            yield {'FrameIndex': self.reader.get_frame_index(),
                   'Hits': hit_series}

    def process(self, blob):
        return next(self.blobs)

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        return next(self.blobs)
