# coding=utf-8
# Filename: dataclasses.py
# cython: embedsignature=True
# cython: linetrace=True
# cython: profile=True
# distutils: define_macros=CYTHON_TRACE=1
# pylint: disable=W0232,C0103,C0111
"""
...

"""
from __future__ import division, absolute_import, print_function

from collections import namedtuple, defaultdict
import ctypes
from libcpp cimport bool as c_bool  # noqa
from six import with_metaclass
from struct import Struct, calcsize

import numpy as np
from numpy.lib import recfunctions as rfn
cimport numpy as np
cimport cython

np.import_array()

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"
__all__ = (
    'Table',)

DEFAULT_H5LOC = '/misc'

TEMPLATE_H5LOCS = {
    'EventInfo': '/event_info',
    'HitSeries': '/hits',
    'CMcHitSeries': '/mc_hits',
    'McHitSeries': '/mc_hits',
    'McTrackSeries': '/mc_tracks',
    'RawHitSeries': '/raw_hits',
    'CRawHitSeries': '/hits',
    'SummaryFrameSeries': '/summary_frame_series',
    'SummaryFrameInfo': '/summary_slice_info',
    'TimesliceHitSeries': '/time_slice_hits',
}

TEMPLATE_DTYPES = {
    'SummaryFrameInfo': np.dtype([
        ('dom_id', '<u4'),
        ('fifo_status', '<u4'),
        ('frame_id', '<u4'),
        ('frame_index', '<u4'),
        ('has_udp_trailer', '<u4'),
        ('high_rate_veto', '<u4'),
        ('max_sequence_number', '<u4'),
        ('n_packets', '<u4'),
        ('slice_id', '<u4'),
        ('utc_nanoseconds', '<u4'),
        ('utc_seconds', '<u4'),
        ('white_rabbit_status', '<u4'),
    ]),
    'EventInfo': np.dtype([
        ('det_id', '<i4'),
        ('frame_index', '<u4'),
        ('livetime_sec', '<u8'),
        ('mc_id', '<i4'),
        ('mc_t', '<f8'),
        ('n_events_gen', '<u8'),
        ('n_files_gen', '<u8'),
        ('overlays', '<u4'),
        ('trigger_counter', '<u8'),
        ('trigger_mask', '<u8'),
        ('utc_nanoseconds', '<u8'),
        ('utc_seconds', '<u8'),
        ('weight_w1', '<f8'),
        ('weight_w2', '<f8'),
        ('weight_w3', '<f8'),
        ('run_id', '<u8'),
        ('event_id', '<u8'),
        ('group_id', '<u4'),
    ]),
    'RawHitSeries': np.dtype([
        ('channel_id', 'u1'),
        ('dom_id', '<u4'),
        ('time', '<f8'),
        ('tot', 'u1'),
        ('triggered', 'u1'),
        ('group_id', '<u4')
    ]),
    'CRawHitSeries': np.dtype([
        ('channel_id', 'u1'),
        ('dir_x', '<f4'),
        ('dir_y', '<f4'),
        ('dir_z', '<f4'),
        ('dom_id', '<u4'),
        ('du', 'u1'),
        ('floor', 'u1'),
        ('pos_x', '<f4'),
        ('pos_y', '<f4'),
        ('pos_z', '<f4'),
        ('t0', '<f4'),
        ('time', '<f8'),
        ('tot', 'u1'),
        ('triggered', 'u1'),
        ('group_id', '<u4'),
    ]),
    'McHitSeries': np.dtype([
        ('a', 'f4'),
        ('origin', '<u4'),
        ('pmt_id', '<u4'),
        ('time', 'f8'),
        ('group_id', '<u4'),
    ]),
    'CMcHitSeries': np.dtype([
        ('a', 'f4'),
        ('dir_x', '<f4'),
        ('dir_y', '<f4'),
        ('dir_z', '<f4'),
        ('origin', '<u4'),
        ('pmt_id', '<u4'),
        ('pos_x', '<f4'),
        ('pos_y', '<f4'),
        ('pos_z', '<f4'),
        ('time', 'f8'),
        ('group_id', '<u4'),
    ]),
    'HitSeries': np.dtype([
        ('channel_id', 'u1'),
        ('dir_x', '<f8'),
        ('dir_y', '<f8'),
        ('dir_z', '<f8'),
        ('dom_id', '<u4'),
        ('id', '<u4'),
        ('pmt_id', '<u4'),
        ('pos_x', '<f8'),
        ('pos_y', '<f8'),
        ('pos_z', '<f8'),
        ('t0', '<i4'),
        ('time', '<i4'),
        ('tot', 'u1'),
        ('triggered', 'u1'),
        ('group_id', '<u4'),
    ]),
    'TimesliceHitSeries': np.dtype([
        ('channel_id', 'u1'),
        ('dom_id', '<u4'),
        ('time', '<i4'),
        ('tot', 'u1'),
    ]),
    'McTrackSeries': np.dtype([
        ('bjorkeny', '<f8'),
        ('dir_x', '<f8'),
        ('dir_y', '<f8'),
        ('dir_z', '<f8'),
        ('energy', '<f8'),
        ('id', '<u4'),
        ('interaction_channel', '<u4'),
        ('is_cc', '<u4'),
        ('length', '<f8'),
        ('pos_x', '<f8'),
        ('pos_y', '<f8'),
        ('pos_z', '<f8'),
        ('time', '<i4'),
        ('type', '<i4'),
        ('group_id', '<u4'),
    ]),
    'TrackSeries': np.dtype([
        ('bjorkeny', '<f8'),
        ('dir_x', '<f8'),
        ('dir_y', '<f8'),
        ('dir_z', '<f8'),
        ('energy', '<f8'),
        ('id', '<u4'),
        ('interaction_channel', '<u4'),
        ('is_cc', '<u4'),
        ('length', '<f8'),
        ('pos_x', '<f8'),
        ('pos_y', '<f8'),
        ('pos_z', '<f8'),
        ('time', '<i4'),
        ('type', '<i4'),
        ('group_id', '<u4'),
    ]),
    'SummaryframeSeries': np.dtype([
        ('dom_id', '<u4'),
        ('max_sequence_number', '<u4'),
        ('n_received_packets', '<u4'),
        ('group_id', '<u4'),
    ]),
}


class Table(np.recarray):
    """2D generic Table with grouping index.

    This class adds the following to ``np.recarray``:

    Attributes
    ----------
    h5loc: str, default='{}'
        HDF5 group where to write into.
    """.format(DEFAULT_H5LOC)

    def __new__(cls, data, h5loc=DEFAULT_H5LOC, dtype=None, **kwargs):
        if isinstance(data, dict):
            return cls.from_dict(data, h5loc=h5loc, dtype=dtype, **kwargs)
        print('from array')
        obj = np.asanyarray(data, dtype=dtype).view(cls)
        obj.h5loc = h5loc
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            # called from explicit contructor
            return obj
        # views or slices
        self.h5loc = getattr(obj, 'h5loc', DEFAULT_H5LOC)
        # attribute access returns void instances on slicing/iteration
        # kudos to https://github.com/numpy/numpy/issues/3581#issuecomment-108957200
        if obj is not None and type(obj) is not type(self):
            self.dtype = np.dtype((np.record, obj.dtype))

    @staticmethod
    def _expand_scalars(map):
        scalars = []
        maxlen = 0
        for k, v in map.items():
            if np.isscalar(v):
                scalars.append(k)
            elif len(v) > maxlen:
                maxlen = len(v)
        for s in scalars:
            map[s] = np.full(maxlen, map[s])
        return map

    @classmethod
    def from_dict(cls, map, dtype=None, **kwargs):
        print('from dict')
        # i hope order of keys == order or values
        if dtype is None:
            names = list(map.keys())
        else:
            dtype = np.dtype(dtype)
            dt_names = [f for f in dtype.names]
            map_names = [k for k in map.keys()]
            if not set(dt_names) == set(map_names):
                raise KeyError('Dictionary keys and dtype fields do not match!')
            names = list(dtype.names)

        map = cls._expand_scalars(map)
        return cls(np.rec.fromarrays(map.values(), names=names,
                                     dtype=dtype), **kwargs)

    @classmethod
    def from_template(cls, data, template):
        dt = TEMPLATE_DTYPES[template]
        loc = TEMPLATE_H5LOCS[template]
        return cls(data, h5loc=loc, dtype=dt)

    def append_fields(self, fields, values, **kwargs):
        """Uses `numpy.lib.recfunctions.append_fields` to append new fields."""
        new_arr = rfn.append_fields(self, fields, values,
                                    usemask=False, **kwargs)
        return self.__class__(new_arr, h5loc=self.h5loc)

    def sorted(self, by):
        """Sort array by a column.

        Parameters
        ==========
        by: str
            Name of the columns (e.g. 'time').
        """
        sort_idc = np.argsort(self[by])
        return self.__class__(self[sort_idc])
