#!/usr/bin/env python2
# Filename: numpy.py
# pylint: disable=locally-disabled
# vim:set ts=4 sts=4 sw=4 et:
"""Streamer for Numpy arrays.
"""
from __future__ import division, absolute_import, print_function

import numpy as np

from km3pipe.core import Pump, Blob


class NumpyStructuredPump(Pump):
    """NumpyStructuredPump streams events from structured Numpy arrays.
    """

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)

        self.array = self.get("array")
        columns = self.get("columns")

        if columns:
            self.columns = columns
        else:
            self.columns = self.array.dtype.names

    def get_blob(self, index):
        blob = Blob()
        sample = self.array[index]
        for key in self.columns:
            blob[key] = sample[key][index]
        return blob

    def __len__(self):
        return self.array.shape[0]

    def next(self):
        return self.__next__()

    def __next__(self):
        try:
            blob = self.get_blob(self.index)
        except IndexError:
            self.index = 0
            raise StopIteration
        self.index += 1
        return blob

    def __iter__(self):
        return self

    def __getitem__(self,index):
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
