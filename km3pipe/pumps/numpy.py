#!/usr/bin/env python2
# Filename: numpy.py
# pylint: disable=locally-disabled
# vim:set ts=4 sts=4 sw=4 et:
""" Pump for Numpy arrays.
"""
from __future__ import division, absolute_import, print_function

import numpy as np

from km3pipe.core import Pump


class NumpyPump(Pump):
    """A pump for Numpy structured arrays.
    """
    def __init__(self, columns=None, **context):
        super(self.__class__, self).__init__(**context)

        self.filename = self.get("filename")
        columns = self.get("columns")

        with np.load(self.filename) as array:
            self.array = array
        if columns:
            self.columns = columns
        else:
            self.columns = self.array.dtype.names

    def blob(self, index):
        blob = {}
        sample = self.array[index]
        for key in self.columns:
            blob[key] = sample[key][index]
        return blob
