#!/usr/bin/env python2
# Filename: numpy.py
# pylint: disable=locally-disabled
# vim:set ts=4 sts=4 sw=4 et:
"""Streamer for Numpy arrays.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import tables

from km3pipe.core import Pump, Blob, Module


class HDF5Loader():
    def __init__(self, filename, where='/', keys=None, ftitle='Data'):
        super(self.__class__, self).__init__(**context)
        self.filename = filename
        self.where = where
        self.ftitle = ftitle
        self.h5file = self._open_h5file(filename)

        self.keys = keys
        self.dsets = {}
        for key in self.keys:
            self.dsets[key] = self._load_array(self.h5file, key, where)

        # TODO
        # read one name header
        # view as float/whatever
        # column_stack
        # view as 2dim recarray(?)
        # data.view(dtype=[(n, 'float64') for n in csv_names]).reshape(len(data))
        # use np_utils

    def get_array(self):
        return self.array

	def _open_h5file(h5file, fmode='r', ftitle='Data'):
		"""Open file if name is given, else pass through."""
		if isinstance(h5file, string_types):
			return tables.open_file(filename=h5file, mode=fmode, ftitle='Data')
		return h5file

	def _load_array(h5file, key, tab_where='/', return_rec=True):
		h5file = self._open_h5file(h5file)
		rec = h5file.get_node(tab_where + key).read()
		if return_rec:
			return rec
		arr = rec[key]
		return arr


class NPYLoader():
    def __init__(self, filename):
        self.filename = filename
        self.array = np.load(filename, 'r',)

    def get_array(self):
        return self.array


class NumpyStructuredPump(Pump):
    """NumpyStructuredPump streams events from structured Numpy arrays.
    """

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.callback = self.get("callback")
        if self.callback:
            self.array = self.callback.get_array()
        else:
            self.array = self.get("array")

        self.columns = self.get("columns")
        if not self.columns:
            self.columns = self.array.dtype.names
        self.index = 0
        self.n_evts = self.array.shape[0]

    def process(self, blob=None):
        try:
            blob = self.get_blob(self.index)
        except IndexError:
            raise StopIteration
        self.index += 1
        return blob

    def get_blob(self, index):
        if index >= self.n_evts:
            raise IndexError
        blob = Blob()
        for key in self.columns:
            # 0 index due to recarray magick
            blob[key] = self.array[key][index]
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
