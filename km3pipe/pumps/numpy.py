#!/usr/bin/env python2
# Filename: numpy.py
# pylint: disable=locally-disabled
# vim:set ts=4 sts=4 sw=4 et:
"""Streamer for Numpy arrays.
"""
from __future__ import division, absolute_import, print_function
from six import string_types

import numpy as np
import tables

from km3pipe.core import Pump, Blob


class HDF5Loader():
    """Open an HDF5 File and store it as structured Numpy array.

    Paramters
    ---------
    h5file: str or tables.File instance
        Name of the HDF5 file to open, or a pytables File instance.
    keys: list of str, optional (default=None)
        The names of the tables to fetch (without '/.../' prefix).
    where: str or dict of strings->strings, optional (default="/")
        Location of the tables inside the file. If a dictionary is passed,
        the location of each key is read as where[key].

    Attributes
    ----------
    array: numpy structured array
        The data retrived from the HDF5 file. The keys are stored as
        columns.

    """
    def __init__(self, h5file, keys=None, where='/', ftitle='Data'):
        self._where = where
        if isinstance(where, string_types):
            self._mutliple_locations = True
        self._ftitle = ftitle
        self._h5file = self._open_h5file(self._h5file)

        self._keys = keys
        self._dsets = {}
        for key in self._keys:
            if self._multiple_locations:
                where = self._where[key]
            else:
                where = self._where
            self._dsets[key] = self._load_array(self._h5file, key, where)
        self.array = np.column_stack(list(self._dsets.values()))
        self.array = self.array.view(
            dtype=[(k, 'float64') for k in self._keys]
        ).reshape(len(self.array))

    def get_array(self):
        """Return the stored array.
        """
        return self.array

    def _open_h5file(self, h5file, fmode='r', ftitle='Data'):
        """Open HDF5 file if string is passed, else do nothing.
        """
        if isinstance(h5file, string_types):
            return tables.open_file(filename=h5file, mode=fmode, ftitle='Data')
        return h5file

    def _load_array(self, h5file, key, where='/', return_rec=False):
        """Retrieve a numpy array from an hdf5 file.
        """
        h5file = self._open_h5file(h5file)
        rec = h5file.get_node(where + key).read()
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

    This pump is different from most file pumps, since it does not open
    files on its own. It rather streams an array returned from an object
    via the `callback` parameter.

        pipe.attach(NumpyStructuredPump, callback=NPYLoader('myfile'))

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
