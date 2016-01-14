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
