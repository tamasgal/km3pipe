# coding=utf-8
# Filename: dataclasses.py
# pylint: disable=W0232,C0103,C0111
# vim:set ts=4 sts=4 sw=4 et syntax=python:
"""
Dataclasses for internal use. Heavily based on Numpy arrays.
"""
from __future__ import division, absolute_import, print_function

from six import string_types

import numpy as np
from numpy.lib import recfunctions as rfn

from .dataclass_templates import TEMPLATES


__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"
__all__ = ('Table', 'is_structured', 'has_structured_dt', 'inflate_dtype')

DEFAULT_H5LOC = '/misc'


def has_structured_dt(arr):
    """Check if the array representation has a structured dtype."""
    arr = np.asanyarray(arr)
    return is_structured(arr.dtype)


def is_structured(dt):
    """Check if the dtype is structured."""
    if not hasattr(dt, 'fields'):
        return False
    return not (dt.fields is None)


def inflate_dtype(arr, names):
    """Create structured dtype from a 2d ndarray with unstructured dtype."""
    arr = np.asanyarray(arr)
    if has_structured_dt(arr):
        return arr.dtype
    s_dt = arr.dtype
    dt = [(n, s_dt) for n in names]
    dt = np.dtype(dt)
    return dt


class Table(np.recarray):
    """2D generic Table with grouping index.

    This class adds the following to ``np.recarray``:

    Parameters
    ----------
    data: array-like or dict(array-like)
        numpy array with structured/flat dtype, or dict of arrays.
    h5loc: str
    Location in HDF5 file where to store the data. [default: '/misc'
    dtype: numpy dtype
        Datatype over array. If not specified and data is an unstructured
        array, ``names`` needs to be specified. [default: None]
    colnames: list(str)
        Column names to use when generating a dtype from unstructured arrays.
        [default: None]

    Attributes
    ----------
    h5loc: str
        HDF5 group where to write into. (default='/misc')
    split_h5: bool
        Split the array into separate arrays, column-wise, when saving
        to hdf5? (default=False)

    Methods
    -------
    from_dict(arr_dict, dtype=None, **kwargs)
        Create an Table from a dict of arrays (similar to pandas).
    from_template(data, template, **kwargs)
        Create an array from a dict of arrays with a predefined dtype.
    sorted(by)
        Sort the table by one of its columns.
    append_columns(colnames, values)
        Append new columns to the table.
    to_dataframe()
        Return as pandas dataframe.
    from_dataframe(df, h5loc)
        Instantiate from a dataframe.
    """

    def __new__(cls, data, h5loc=DEFAULT_H5LOC, dtype=None,
                colnames=None, split_h5=False, **kwargs):
        if isinstance(data, dict):
            return cls.from_dict(data, h5loc=h5loc, dtype=dtype,
                                 split_h5=split_h5, colnames=colnames, **kwargs)
        if isinstance(data, list) or isinstance(data, tuple):
            return cls.from_list(data, h5loc=h5loc, dtype=dtype,
                                 split_h5=split_h5, colnames=colnames, **kwargs)
        if not has_structured_dt(data):
            # flat (nonstructured) dtypes fail miserably!
            # default to `|V8` whyever
            if dtype is None or not is_structured(dtype):
                # infer structured dtype from array data + column names
                if colnames is None:
                    raise ValueError(
                        "Need to either specify column names or a "
                        "structured dtype when passing unstructured arrays!"
                    )
                dtype = inflate_dtype(data, colnames)
            # this *should* have been checked above, but do this
            # just to be sure in case I screwed up the logic above;
            # users will never see this, this should only show in tests
            assert is_structured(dtype)
            data = np.asanyarray(data).view(dtype)
            dtype = data.dtype

        obj = np.asanyarray(data, dtype=dtype).view(cls)
        obj.h5loc = h5loc
        obj.split_h5 = split_h5
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            # called from explicit contructor
            return obj
        # views or slices
        self.h5loc = getattr(obj, 'h5loc', DEFAULT_H5LOC)
        self.split_h5 = getattr(obj, 'split_h5', False)
        # attribute access returns void instances on slicing/iteration
        # kudos to https://github.com/numpy/numpy/issues/3581#issuecomment-108957200
        if obj is not None and type(obj) is not type(self):
            self.dtype = np.dtype((np.record, obj.dtype))

    @staticmethod
    def _expand_scalars(arr_dict):
        scalars = []
        maxlen = 1      # have at least 1-elem arrays
        for k, v in arr_dict.items():
            if np.isscalar(v):
                scalars.append(k)
            elif len(v) > maxlen:
                maxlen = len(v)
        for s in scalars:
            arr_dict[s] = np.full(maxlen, arr_dict[s])
        return arr_dict

    @classmethod
    def from_dict(cls, arr_dict, dtype=None, **kwargs):
        """Generate a table from a dictionary of arrays.
        """
        # i hope order of keys == order or values
        if dtype is None:
            names = list(arr_dict.keys())
        else:
            dtype = np.dtype(dtype)
            dt_names = [f for f in dtype.names]
            dict_names = [k for k in arr_dict.keys()]
            if not set(dt_names) == set(dict_names):
                raise KeyError('Dictionary keys and dtype fields do not match!')
            names = list(dtype.names)

        arr_dict = cls._expand_scalars(arr_dict)
        return cls(np.rec.fromarrays(arr_dict.values(), names=names,
                                     dtype=dtype), **kwargs)

    @classmethod
    def from_list(cls, arr_list, dtype=None, colnames=None, **kwargs):
        if dtype is None or not is_structured(dtype):
            # infer structured dtype from array data + column names
            if colnames is None:
                raise ValueError(
                    "Need to either specify column names or a "
                    "structured dtype when passing unstructured arrays!"
                )
            dtype = inflate_dtype(arr_list, colnames)
            colnames = dtype.names
        print(dtype)
        print(dtype.names)
        print(colnames)
        print(arr_list)
        if len(arr_list) != len(dtype.names):
            raise ValueError(
                "Number of columns mismatch between data and dtype!")
        return cls(
            {k: arr_list[i] for i, k in enumerate(dtype.names)},
            dtype=dtype, colnames=colnames, **kwargs)

    @property
    def templates_avail(self):
        return sorted(list(TEMPLATES.keys()))

    @classmethod
    def from_template(cls, data, template_name):
        """Create a table from a predefined datatype.

        See the ``templates_avail`` property for available names.

        Parameters
        ----------
        data
            Data in a format that the ``__init__`` understands.
        template: str
            Name of the dtype template to use.
        """
        template = TEMPLATES[template_name]
        dt = template['dtype']
        loc = template['h5loc']
        split = template['split_h5']
        return cls(data, h5loc=loc, dtype=dt, split_h5=split)

    @staticmethod
    def _check_column_length(colnames, values, n):
        values = np.atleast_2d(values)
        for i in range(len(values)):
            v = values[i]
            if len(v) == n:
                continue
            else:
                if len(values[i]) != n:
                    raise ValueError(
                        "Trying to append more than one column, but "
                        "some arrays mismatch in length!")

    def append_columns(self, colnames, values, **kwargs):
        """Append new columns to the table.

        When appending a single column, ``values`` can be a scalar or an
        array of either length 1 or the same length as this array (the one
        it's appended to). In case of multiple columns, values must have
        the shape ``list(arrays)``, and the dimension of each array
        has to match the length of this array.

        See the docs for ``numpy.lib.recfunctions.append_fields`` for an
        explanation of the remaining options.
        """
        n = len(self)
        if np.isscalar(values):
            values = np.full(n, values)

        values = np.atleast_1d(values)
        if not isinstance(colnames, string_types) and len(colnames) > 1:
            values = np.atleast_2d(values)
            self._check_column_length(colnames, values, n)

        if values.ndim == 1:
            if len(values) > n:
                raise ValueError(
                    "New Column is longer than existing table!")
            elif len(values) > 1 and len(values) < n:
                raise ValueError(
                    "New Column is shorter than existing table, "
                    "but not just one element!")
            elif len(values) == 1:
                values = np.full(n, values[0])
        new_arr = rfn.append_fields(self, colnames, values,
                                    usemask=False, **kwargs)
        return self.__class__(new_arr, h5loc=self.h5loc,
                              split_h5=self.split_h5)

    def sorted(self, by, **kwargs):
        """Sort array by a column.

        Parameters
        ==========
        by: str
            Name of the columns to sort by(e.g. 'time').
        """
        sort_idc = np.argsort(self[by], **kwargs)
        return self.__class__(self[sort_idc], h5loc=self.h5loc,
                              split_h5=self.split_h5)

    def to_dataframe(self):
        from pandas import DataFrame
        return DataFrame(self)

    @classmethod
    def from_dataframe(cls, df, h5loc=DEFAULT_H5LOC, split_h5=False):
        rec = df.to_records(index=False)
        return cls(rec, h5loc=h5loc, split_h5=split_h5)

    def __str__(self):
        s = "HDF5 location: {}\n".format(self.h5loc)
        s += "\n".join(map(lambda d: "{} (dtype: {}) = {}"
                                     .format(*d, self[d[0]]),
                           self.dtype.descr))
        return s
