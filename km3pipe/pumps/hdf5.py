# coding=utf-8
# Filename: hdf5.py
# pylint: disable=C0103,R0903
"""
Pumps for the EVT simulation dataformat.

"""
from __future__ import division, absolute_import, print_function

import os.path
import pandas as pd

from km3pipe import Pump
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = 'tamasgal'


class HDF5Pump(Pump):
    """Provides a pump for KM3NeT HDF5 files"""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename')
        if os.path.isfile(self.filename):
            self._store = pd.HDFStore(self.filename)
            self._hits_by_event = self._store.hits.groupby('event_id')
            self._n_events = len(self._hits_by_event)
        else:
            raise IOError("No such file or directory: '{0}'"
                          .format(self.filename))
        self.index = 0

    def process(self, blob):
        try:
            blob = self.get_blob(self.index)
        except KeyError:
            self.index = 0
            raise StopIteration
        self.index += 1
        return blob

    def get_blob(self, index):
        hits = self._hits_by_event.get_group(index)
        blob = {'HitTable': hits}
        return blob

    def finish(self):
        """Clean everything up"""
        self._store.close()

    def __len__(self):
        return self._n_events

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        try:
            blob = self.get_blob(self.index)
        except KeyError:
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
