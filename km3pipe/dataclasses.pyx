# coding=utf-8
# cython: profile=True
# Filename: dataclasses.py
# cython: embedsignature=True
# pylint: disable=W0232,C0103,C0111
"""
...

"""
from __future__ import division, absolute_import, print_function

import ctypes
from libcpp cimport bool as c_bool  # noqa
from six import with_metaclass

import numpy as np
cimport numpy as np
cimport cython
import pandas as pd

np.import_array()

from km3pipe.tools import angle_between, geant2pdg, pdg2name

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"
__all__ = ('EventInfo', 'Point', 'Position', 'Direction', 'HitSeries',
           'TimesliceHitSeries', 'Hit',
           'Track', 'TrackSeries', 'Serialisable', 'SummaryframeInfo')


IS_CC = {
    3: 0,         # False,
    2: 1,         # True,
    1: 0,         # False,
    0: 1,         # True,
}


class Serialisable(type):
    """A metaclass for serialisable classes.

    The classes should define a `dtype` attribute in their body and are not
    meant to define `__init__` (it will be overwritten).

    Example using six.with_metaclass for py2/py3 compat
    ---------------------------------------------------

        class Foo(with_metaclass(Serialisable)):
            dtype = np.dtype([('a', '<i4'), ('b', '>i8')])

    """
    def __new__(metaclass, class_name, class_parents, class_attr):
        attr = {'h5loc': '/'}
        for name, val in class_attr.items():
            if name == 'dtype':
                attr['dtype'] = np.dtype(val)
            else:
                attr[name] = val

        def __init__(self, *args, **kwargs):
            """Take care of the attribute settings."""
            for arg, name in zip(args, self.dtype.names):
                setattr(self, name, arg)
            for key, value in kwargs.iteritems():
                setattr(self, key, value)

        attr['__init__'] = __init__

        return type(class_name, class_parents, attr)


class SummarysliceInfo(with_metaclass(Serialisable)):
    """JDAQSummaryslice Metadata.
    """
    h5loc = '/'
    dtype = np.dtype([
        ('det_id', '<i4'),
        ('frame_index', '<u4'),
        ('run_id', '<i4'),
        ('slice_id', '<u4'),
    ])

    @classmethod
    def from_table(cls, row):
        args = []
        for col in cls.dtype.names:
            try:
                args.append(row[col])
            except KeyError:
                args.append(np.nan)
        return cls(*args)

    @classmethod
    def deserialise(cls, *args, **kwargs):
        return cls.conv_from(*args, **kwargs)

    def serialise(self, *args, **kwargs):
        return self.conv_to(*args, **kwargs)

    @classmethod
    def conv_from(cls, data, event_id, fmt='numpy', h5loc='/'):
        if fmt == 'numpy':
            return cls.from_table(data[0])

    def conv_to(self, to='numpy'):
        if to == 'numpy':
            return KM3Array(np.array(self.__array__(), dtype=self.dtype),
                            h5loc=self.h5loc)
        if to == 'pandas':
            return KM3DataFrame(self.conv_to(to='numpy'), h5loc=self.h5loc)

    def __array__(self):
        return [(
            self.det_id, self.frame_index, self.run_id, self.slice_id,
        ), ]

    def __str__(self):
        return "Summaryslice #{0}:\n" \
               "    detector id:     {1}\n" \
               "    frame index:     {2}\n" \
               "    run id:          {3}\n" \
               .format(self.slice_id,
                       self.det_id, self.frame_index, self.run_id,
                       )

    def __insp__(self):
        return self.__str__()

    def __len__(self):
        return 1


class TimesliceInfo(with_metaclass(Serialisable)):
    """JDAQTimeslice metadata.
    """
    h5loc = '/'
    dtype = np.dtype([
        ('dom_id', '<u4'),
        ('frame_id', '<u4'),
        ('n_hits', '<u4'),
        ('slice_id', '<u4'),
    ])

    @classmethod
    def from_table(cls, row):
        args = []
        for col in cls.dtype.names:
            try:
                args.append(row[col])
            except KeyError:
                args.append(np.nan)
        return cls(*args)

    @classmethod
    def deserialise(cls, *args, **kwargs):
        return cls.conv_from(*args, **kwargs)

    def serialise(self, *args, **kwargs):
        return self.conv_to(*args, **kwargs)

    @classmethod
    def conv_from(cls, data, frame_id, fmt='numpy', h5loc='/'):
        if fmt == 'numpy':
            return cls.from_table(data[0])

    def conv_to(self, to='numpy'):
        if to == 'numpy':
            return KM3Array(np.array(self.__array__(), dtype=self.dtype),
                            h5loc=self.h5loc)
        if to == 'pandas':
            return KM3DataFrame(self.conv_to(to='numpy'), h5loc=self.h5loc)

    def __array__(self):
        return [(
            self.dom_id, self.frame_id, self.n_hits, self.slice_id,
        ), ]

    def __str__(self):
        return "Timeslice frame:\n" \
               "    slice id: {0}\n" \
               "    frame id: {1}\n" \
               "    DOM id:   {2}\n" \
               "    n_hits:   {3}\n" \
               .format(self.slice_id, self.frame_id, self.dom_id, self.n_hits)

    def __insp__(self):
        return self.__str__()

    def __len__(self):
        return 1


class TimesliceFrameInfo(with_metaclass(Serialisable)):
    """JDAQTimeslice frame metadata.
    """
    h5loc = '/'
    dtype = np.dtype([
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
    ])

    @classmethod
    def from_table(cls, row):
        args = []
        for col in cls.dtype.names:
            try:
                args.append(row[col])
            except KeyError:
                args.append(np.nan)
        return cls(*args)

    @classmethod
    def deserialise(cls, *args, **kwargs):
        return cls.conv_from(*args, **kwargs)

    def serialise(self, *args, **kwargs):
        return self.conv_to(*args, **kwargs)

    @classmethod
    def conv_from(cls, data, frame_id, fmt='numpy', h5loc='/'):
        if fmt == 'numpy':
            return cls.from_table(data[0])

    def conv_to(self, to='numpy'):
        if to == 'numpy':
            return KM3Array(np.array(self.__array__(), dtype=self.dtype),
                            h5loc=self.h5loc)
        if to == 'pandas':
            return KM3DataFrame(self.conv_to(to='numpy'), h5loc=self.h5loc)

    def __array__(self):
        return [(
            self.dom_id, self.fifo_status, self.frame_id, self.frame_index,
            self.has_udp_trailer, self.high_rate_veto,
            self.max_sequence_number, self.n_packets, self.slice_id,
            self.utc_nanoseconds, self.utc_seconds, self.white_rabbit_status
        ), ]

    def __str__(self):
        return "Timeslice frame:\n" \
               "    slice id: {0}\n" \
               "    frame id: {1}\n" \
               "    DOM id:   {2}\n" \
               "    UDP packets: {3}/{4}\n" \
               .format(self.slice_id, self.frame_id, self.dom_id,
                       self.n_packets,
                       self.max_sequence_number)

    def __insp__(self):
        return self.__str__()

    def __len__(self):
        return 1


class SummaryframeInfo(with_metaclass(Serialisable)):
    """JDAQSummaryslice frame metadata.
    """
    h5loc = '/'
    dtype = np.dtype([
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
    ])

    @classmethod
    def from_table(cls, row):
        args = []
        for col in cls.dtype.names:
            try:
                args.append(row[col])
            except KeyError:
                args.append(np.nan)
        return cls(*args)

    @classmethod
    def deserialise(cls, *args, **kwargs):
        return cls.conv_from(*args, **kwargs)

    def serialise(self, *args, **kwargs):
        return self.conv_to(*args, **kwargs)

    @classmethod
    def conv_from(cls, data, frame_id, fmt='numpy', h5loc='/'):
        if fmt == 'numpy':
            return cls.from_table(data[0])

    def conv_to(self, to='numpy'):
        if to == 'numpy':
            return KM3Array(np.array(self.__array__(), dtype=self.dtype),
                            h5loc=self.h5loc)
        if to == 'pandas':
            return KM3DataFrame(self.conv_to(to='numpy'), h5loc=self.h5loc)

    def __array__(self):
        return [(
            self.dom_id, self.fifo_status, self.frame_id, self.frame_index,
            self.has_udp_trailer, self.high_rate_veto,
            self.max_sequence_number, self.n_packets, self.slice_id,
            self.utc_nanoseconds, self.utc_seconds, self.white_rabbit_status
        ), ]

    def __str__(self):
        return "Summaryslice frame #{0}:\n" \
               "    slice id:    {1}\n" \
               "    DOM id:      {2}\n" \
               "    UDP packets: {3}/{4}\n" \
               .format(self, self.frame_id, self.slice_id, self.dom_id,
                       self.n_packets,
                       self.max_sequence_number)

    def __insp__(self):
        return self.__str__()

    def __len__(self):
        return 1


class EventInfo(object):
    """Event Metadata.
    """
    dtype = np.dtype([
        ('det_id', '<i4'),
        ('frame_index', '<u4'),
        ('livetime_sec', '<u8'),
        ('mc_id', '<i4'),
        ('mc_t', '<f8'),
        ('n_events_gen', '<u8'),
        ('n_files_gen', '<u8'),
        ('overlays', '<u4'),
        # ('run_id', '<u4'),
        ('trigger_counter', '<u8'),
        ('trigger_mask', '<u8'),
        ('utc_nanoseconds', '<u8'),
        ('utc_seconds', '<u8'),
        ('weight_w1', '<f8'),
        ('weight_w2', '<f8'),
        ('weight_w3', '<f8'),
        ('event_id', '<u4'),
    ])

    def __init__(self, arr, h5loc='/'):
        self._arr = np.array(arr, dtype=self.dtype).reshape(1)
        for col in self.dtype.names:
            setattr(self, col, self._arr[col])
        self.h5loc = h5loc

    @classmethod
    def from_row(cls, row, **kwargs):
        args = tuple((row[col] for col in cls.dtype.names))
        return cls(np.array(args, dtype=cls.dtype), **kwargs)

    @classmethod
    def deserialise(cls, *args, **kwargs):
        return cls.conv_from(*args, **kwargs)

    def serialise(self, *args, **kwargs):
        return self.conv_to(*args, **kwargs)

    @classmethod
    def conv_from(cls, data, event_id, fmt='numpy', h5loc='/', **kwargs):
        if fmt == 'numpy':
            return cls.from_row(data, **kwargs)

    def conv_to(self, to='numpy', **kwargs):
        if to == 'numpy':
            return KM3Array(np.array(self.__array__(), dtype=self.dtype),
                            h5loc=self.h5loc)
        if to == 'pandas':
            return KM3DataFrame(self.conv_to(to='numpy'), h5loc=self.h5loc,
                                **kwargs)

    def __array__(self):
        return self._arr

    def __str__(self):
        return "Event #{0}:\n" \
               "    detector id:     {1}\n" \
               "    frame index:     {2}\n" \
               "    UTC seconds:     {3}\n" \
               "    UTC nanoseconds: {4}\n" \
               "    MC id:           {5}\n" \
               "    MC time:         {6}\n" \
               "    overlays:        {7}\n" \
               "    trigger counter: {8}\n" \
               "    trigger mask:    {9}" \
               .format(self.event_id, self.det_id,
                       self.frame_index, self.utc_seconds,
                       self.utc_nanoseconds, self.mc_id, self.mc_t,
                       self.overlays, self.trigger_counter, self.trigger_mask,
                       # self.run_id,
                       )

    def __repr__(self):
        return self.__str__()

    def __insp__(self):
        return self.__str__()

    def __len__(self):
        return 1


class Point(np.ndarray):
    """Represents a point in a 3D space"""
    def __new__(cls, input_array=(np.nan, np.nan, np.nan)):
        """Add x, y and z to the ndarray"""
        obj = np.asarray(input_array).view(cls)
        return obj

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, value):
        self[2] = value


Position = Direction = Point  # Backwards compatibility


class Direction_(Point):
    """Represents a direction in a 3D space

    The direction vector always normalises itself when an attribute is changed.

    """
    def __new__(cls, input_array=(1, 0, 0)):
        """Add x, y and z to the ndarray"""
        normed_array = np.array(input_array) / np.linalg.norm(input_array)
        obj = np.asarray(normed_array).view(cls)
        return obj

    def _normalise(self):
        normed_array = self / np.linalg.norm(self)
        self[0] = normed_array[0]
        self[1] = normed_array[1]
        self[2] = normed_array[2]

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value
        self._normalise()

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value
        self._normalise()

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, value):
        self[2] = value
        self._normalise()

    @property
    def zenith(self):
        return angle_between(self, (0, 0, -1))

    def __str__(self):
        return "({0:.4}, {1:.4}, {2:.4})".format(self.x, self.y, self.z)


cdef class Hit:
    """Hit on a PMT.

    Parameters
    ----------
    channel_id : int
    dir_x : float
    dir_y : float
    dir_z : float
    dom_id : int
    id : int
    pmt_id : int
    pos_x : float
    pos_y : float
    pos_z : float
    t0 : int
    time : int
    tot : int
    triggered : int

    """
    cdef public int id, dom_id, time, tot, channel_id, pmt_id
    cdef public unsigned short int triggered
    cdef public float pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, t0

    def __cinit__(self,
                  int channel_id,
                  float dir_x,
                  float dir_y,
                  float dir_z,
                  int dom_id,
                  int id,
                  int pmt_id,
                  float pos_x,
                  float pos_y,
                  float pos_z,
                  int t0,
                  int time,
                  int tot,
                  unsigned short int triggered,
                  int event_id=0        # ignore this!
                  ):
        self.channel_id = channel_id
        self.dir_x = dir_x
        self.dir_y = dir_y
        self.dir_z = dir_z
        self.dom_id = dom_id
        self.id = id
        self.pmt_id = pmt_id
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.t0 = t0
        self.time = time
        self.tot = tot
        self.triggered = triggered

    @property
    def dir(self):
        return np.array((self.dir_x, self.dir_y, self.dir_z))

    @property
    def pos(self):
        return np.array((self.pos_x, self.pos_y, self.pos_z))

    def __str__(self):
        return "Hit: channel_id({0}), dom_id({1}), pmt_id({2}), tot({3}), " \
               "time({4}), triggered({5})" \
               .format(self.channel_id, self.dom_id, self.pmt_id, self.tot,
                       self.time, self.triggered)

    def __repr__(self):
        return self.__str__()

    def __insp__(self):
        return self.__str__()


cdef class TimesliceHit:
    """Timeslice hit on a PMT.

    Parameters
    ----------
    channel_id : int
    dom_id : int
    time : int
    tot : int

    """
    cdef public int dom_id, time, tot, channel_id
    cdef public np.ndarray pos
    cdef public np.ndarray dir

    def __cinit__(self,
                  int channel_id,
                  int dom_id,
                  int time,
                  int tot,
                  int frame_id=0        # ignore this!
                  ):
        self.channel_id = channel_id
        self.dom_id = dom_id
        self.time = time
        self.tot = tot

    def __str__(self):
        return "TimesliceHit: channel_id({0}), dom_id({1}), tot({2}), " \
               "time({3})" \
               .format(self.channel_id, self.dom_id, self.tot, self.time)

    def __repr__(self):
        return self.__str__()

    def __insp__(self):
        return self.__str__()


cdef class MCHit:
    """Monte Carlo Hit on a PMT.

    Parameters
    ----------
    channel_id : int
    dir : Direction or numpy.ndarray
    dom_id : int
    id : int
    pmt_id : int
    pos : Position or numpy.ndarray
    time : int
    tot : int
    triggered : int

    """
    cdef public int id, dom_id, time, tot, channel_id, pmt_id
    cdef public unsigned short int triggered
    cdef public np.ndarray pos
    cdef public np.ndarray dir

    def __cinit__(self,
                  int channel_id,
                  int dom_id,
                  int id,
                  int pmt_id,
                  int time,
                  int tot,
                  unsigned short int triggered,
                  ):
        self.channel_id = channel_id
        self.dom_id = dom_id
        self.id = id
        self.pmt_id = pmt_id
        self.time = time
        self.tot = tot
        self.triggered = triggered

    def __str__(self):
        return "Hit: channel_id({0}), dom_id({1}), pmt_id({2}), tot({3}), " \
               "time({4}), triggered({5})" \
               .format(self.channel_id, self.dom_id, self.pmt_id, self.tot,
                       self.time, self.triggered)

    def __repr__(self):
        return self.__str__()

    def __insp__(self):
        return self.__str__()


cdef class Track:
    """Particle track.

    Parameters
    ----------
    bjorkeny : float
    dir : Direction or numpy.ndarray
    energy : float
    id : int
    interaction_channel : int
    is_cc : int
    length : float
    pos : Position or numpy.ndarray
    time : int
    type : int
    """
    cdef public int id, time, type, interaction_channel
    cdef public float energy, length, bjorkeny
    cdef public unsigned short int is_cc
    cdef public np.ndarray pos
    cdef public np.ndarray dir

    def __cinit__(self,
                  float bjorkeny,
                  dir,
                  float energy,
                  int id,
                  np.int64_t interaction_channel,
                  unsigned short int is_cc,
                  float length,
                  pos,
                  int time,
                  int type
                  ):
        self.bjorkeny = bjorkeny
        self.is_cc = is_cc
        self.dir = dir
        self.energy = energy
        self.interaction_channel = interaction_channel
        self.id = id
        self.length = length
        self.pos = pos
        self.time = time
        self.type = type

    def __str__(self):
        return "Track: pos({0}), dir({1}), t={2}, E={3}, type={4} ({5})" \
               .format(self.pos, self.dir, self.time, self.energy,
                       self.type, pdg2name(self.type))

    def __repr__(self):
        return self.__str__()

    def __insp__(self):
        return self.__str__()


class HitSeries(object):
    """Collection of multiple Hits.
    """
    dtype = np.dtype([
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
        # ('run_id', '<u4'),
        ('t0', '<i4'),
        ('time', '<i4'),
        ('tot', 'u1'),
        ('triggered', 'u1'),
        ('event_id', '<u4'),
    ])

    def __init__(self, arr, h5loc='/'):
        self._arr = arr
        self._index = 0
        self._hits = None
        self.h5loc = h5loc

    @classmethod
    def from_aanet(cls, hits, event_id):
        try:
            return cls(np.array([(
                h.channel_id,
                np.nan,     # h.dir.x,
                np.nan,     # h.dir.y,
                np.nan,     # h.dir.z,
                h.dom_id,
                h.id,
                h.pmt_id,
                np.nan,     # h.pos.x,
                np.nan,     # h.pos.y,
                np.nan,     # h.pos.z,
                0,          # t0
                h.t,
                h.tot,
                h.trig,
                event_id,
            ) for h in hits], dtype=cls.dtype))
        except ValueError:
            # Older aanet format.
            return cls(np.array([(
                ord(h.channel_id),
                np.nan,     # h.dir.x,
                np.nan,     # h.dir.y,
                np.nan,     # h.dir.z,
                h.dom_id,
                h.id,
                h.pmt_id,
                np.nan,     # h.pos.x,
                np.nan,     # h.pos.y,
                np.nan,     # h.pos.z,
                0,          # t0
                h.t,
                h.tot,
                h.trig,
                event_id,
            ) for h in hits], dtype=cls.dtype))

    @classmethod
    def from_evt(cls, hits, event_id):
        return cls(np.array([(
            0,     # channel_id
            np.nan,
            np.nan,
            np.nan,
            0,     # dom_id
            h.id,
            h.pmt_id,
            np.nan,
            np.nan,
            np.nan,
            0,      # t0
            h.time,
            h.tot,
            0,     # triggered
            event_id,
        ) for h in hits], dtype=cls.dtype))

    @classmethod
    def from_arrays(cls, channel_ids, dir_xs, dir_ys, dir_zs, dom_ids, ids,
                    pmt_ids, pos_xs, pos_ys, pos_zs, t0s, times, tots,
                    triggereds, event_id):
        # do we need shape[0] or does len() work too?
        try:
            length = channel_ids.shape[0]
        except AttributeError:
            length = len(channel_ids)
        hits = np.empty(length, cls.dtype)
        hits['channel_id'] = channel_ids
        hits['dir_x'] = dir_xs
        hits['dir_y'] = dir_ys
        hits['dir_z'] = dir_zs
        hits['dom_id'] = dom_ids
        hits['id'] = ids
        hits['pmt_id'] = pmt_ids
        hits['pos_x'] = pos_xs
        hits['pos_y'] = pos_ys
        hits['pos_z'] = pos_zs
        hits['t0'] = t0s
        hits['time'] = times
        hits['tot'] = tots
        hits['triggered'] = triggereds
        hits['event_id'] = np.full(length, event_id, dtype='<u4')
        return cls(hits)

    @classmethod
    def from_dict(cls, map, event_id):
        if event_id is None:
            event_id = map['event_id']
        return cls.from_arrays(
            map['channel_id'],
            map['dir_x'],
            map['dir_y'],
            map['dir_z'],
            map['dom_id'],
            map['id'],
            map['pmt_id'],
            map['pos_x'],
            map['pos_y'],
            map['pos_z'],
            map['t0'],
            map['time'],
            map['tot'],
            map['triggered'],
            event_id,
        )

    @classmethod
    def from_table(cls, table, event_id):
        if event_id is None:
            event_id = table[0]['event_id']
        return cls(np.array([(
            row['channel_id'],
            row['dir_x'],
            row['dir_y'],
            row['dir_z'],
            row['dom_id'],
            row['id'],
            row['pmt_id'],
            row['pos_x'],
            row['pos_y'],
            row['pos_z'],
            row['t0'],
            row['time'],
            row['tot'],
            row['triggered'],
        ) for row in table], dtype=cls.dtype), event_id)

    @classmethod
    def deserialise(cls, *args, **kwargs):
        return cls.conv_from(*args, **kwargs)

    def serialise(self, *args, **kwargs):
        return self.conv_to(*args, **kwargs)

    @classmethod
    def conv_from(cls, data, event_id=None, fmt='numpy', h5loc='/'):
        # what is event_id doing here?
        if fmt == 'numpy':
            return cls(data)
        if fmt == 'pandas':
            return cls(data.to_records(index=False))

    def conv_to(self, to='numpy'):
        if to == 'numpy':
            return KM3Array(np.array(self.__array__(), dtype=self.dtype),
                            h5loc=self.h5loc)
        if to == 'pandas':
            return KM3DataFrame(self.conv_to(to='numpy'), h5loc=self.h5loc)

    def __array__(self):
        return self._arr

    def __iter__(self):
        return self

    @property
    def id(self):
        return self._arr['id']

    @property
    def time(self):
        return self._arr['time']

    @property
    def triggered_hits(self):
        return HitSeries(self._arr[self._arr['triggered'] == True])     # noqa

    @property
    def first_hits(self):
        h = self.conv_to(to='pandas')
        h.sort_values('time', inplace=True)
        h = h.drop_duplicates(subset='dom_id')
        return HitSeries(h.to_records(index=False))

    @property
    def triggered(self):
        return self._arr['triggered']

    @property
    def tot(self):
        return self._arr['tot']

    @property
    def dom_id(self):
        return self._arr['dom_id']

    @property
    def pmt_id(self):
        return self._arr['pmt_id']

    @property
    def id(self):
        return self._arr['id']

    @property
    def channel_id(self):
        return self._arr['channel_id']

    @property
    def pos_x(self):
        return self._arr['pos_x']

    @property
    def pos_y(self):
        return self._arr['pos_y']

    @property
    def pos_z(self):
        return self._arr['pos_z']

    @property
    def dir_x(self):
        return self._arr['dir_x']

    @property
    def dir_y(self):
        return self._arr['dir_y']

    @property
    def dir_z(self):
        return self._arr['dir_z']

    @property
    def t0(self):
        return self._arr['t0']

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        if self._index >= len(self):
            self._index = 0
            raise StopIteration
        hit = self[self._index]
        self._index += 1
        return hit

    def __len__(self):
        return self._arr.shape[0]

    def __getitem__(self, index):
        if isinstance(index, int):
            hit = Hit(*self._arr[index])
            return hit
        elif isinstance(index, slice):
            return self._slice_generator(index)
        else:
            raise TypeError("index must be int or slice")

    def _slice_generator(self, index):
        """A simple slice generator for iterations"""
        start, stop, step = index.indices(len(self))
        for i in range(start, stop, step):
            yield Hit(*self._arr[i])

    def __str__(self):
        n_hits = len(self)
        plural = 's' if n_hits > 1 or n_hits == 0 else ''
        return("HitSeries with {0} hit{1}.".format(len(self), plural))

    def __repr__(self):
        return self.__str__()

    def __insp__(self):
        return '\n'.join([str(hit) for hit in self._hits])


class TimesliceHitSeries(object):
    """Collection of multiple timeslice hits.
    """
    dtype = np.dtype([
        ('channel_id', 'u1'),
        ('dom_id', '<u4'),
        ('time', '<i4'),
        ('tot', 'u1'),
    ])

    def __init__(self, arr, slice_id, frame_id):
        self._arr = arr
        self._index = 0
        self._hits = None
        self.slice_id = slice_id
        self.frame_id = frame_id

    @property
    def h5loc(self):
        return "/timeslices/slice_{0}".format(self.slice_id)

    @property
    def tabname(self):
        return "frame_{0}".format(self.frame_id)

    @classmethod
    def from_arrays(cls, channel_ids, dom_ids, times, tots, slice_id,
                    frame_id):
        len = channel_ids.shape[0]
        hits = np.empty(len, cls.dtype)
        hits['channel_id'] = channel_ids
        hits['dom_id'] = dom_ids
        hits['time'] = times
        hits['tot'] = tots
        return cls(hits, slice_id, frame_id)

    @classmethod
    def from_dict(cls, map, slice_id, frame_id):
        return cls.from_arrays(
            map['channel_id'],
            map['dom_id'],
            map['time'],
            map['tot'],
            slice_id,
            frame_id,
        )

    def from_table(cls, table, slice_id, frame_id):
        return cls(np.array([(
            row['channel_id'],
            row['dom_id'],
            row['time'],
            row['tot'],
        ) for row in table], dtype=cls.dtype), slice_id, frame_id)

    @classmethod
    def deserialise(cls, *args, **kwargs):
        return cls.conv_from(*args, **kwargs)

    def serialise(self, *args, **kwargs):
        return self.conv_to(*args, **kwargs)

    @classmethod
    def conv_from(cls, data, slice_id, frame_id, fmt='numpy', h5loc='/'):
        if fmt == 'numpy':
            # return cls.from_table(data, frame_id)
            return cls(data, slice_id, frame_id)

    def conv_to(self, to='numpy'):
        if to == 'numpy':
            return KM3Array(np.array(self.__array__(), dtype=self.dtype),
                            h5loc=self.h5loc)
        if to == 'pandas':
            return KM3DataFrame(self.conv_to(to='numpy'), h5loc=self.h5loc)

    def __array__(self):
        return self._arr

    def __iter__(self):
        return self

    @property
    def time(self):
        return self._arr['time']

    @property
    def tot(self):
        return self._arr['tot']

    @property
    def dom_id(self):
        return self._arr['dom_id']

    @property
    def channel_id(self):
        return self._arr['channel_id']

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        if self._index >= len(self):
            self._index = 0
            raise StopIteration
        hits = TimesliceHit(*self._arr[self._index])
        self._index += 1
        return hits

    def __len__(self):
        return self._arr.shape[0]

    def __getitem__(self, index):
        if isinstance(index, int):
            return TimesliceHit(*self._arr[index])
        elif isinstance(index, slice):
            return self._slice_generator(index)
        else:
            raise TypeError("index must be int or slice")

    def _slice_generator(self, index):
        """A simple slice generator for iterations"""
        start, stop, step = index.indices(len(self))
        for i in range(start, stop, step):
            yield TimesliceHit(*self._arr[i])

    def __str__(self):
        n_hits = len(self)
        plural = 's' if n_hits > 1 or n_hits == 0 else ''
        return("TimesliceHitSeries with {0} hit{1}.".format(len(self), plural))

    def __repr__(self):
        return self.__str__()

    def __insp__(self):
        return '\n'.join([str(hit) for hit in self._hits])


class TrackSeries(object):
    """Collection of multiple Tracks.

    Attributes
    ----------
    dtype: datatype of array representation
    """
    dtype = np.dtype([
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
        # ('run_id', '<u4'),
        ('time', '<i4'),
        ('type', '<i4'),
        ('event_id', '<u4'),
    ])

    def __init__(self, tracks, event_id, h5loc='/'):
        self._bjorkeny = None
        self._dir = None
        self._energy = None
        self._highest_energetic_muon = None
        self._id = None
        self._index = 0
        self._interaction_channel = None
        self._is_cc = None
        self._length = None
        self._pos = None
        self._time = None
        self._tracks = tracks
        self._type = None
        self.event_id = event_id
        self.h5loc = h5loc

    @classmethod
    def from_aanet(cls, tracks, event_id):
        return cls([Track(cls.get_usr_item(t, 1),               # bjorkeny
                          Direction((t.dir.x, t.dir.y, t.dir.z)),
                          t.E,
                          t.id,
                          cls.get_usr_item(t, 2),               # ichan
                          IS_CC[cls.get_usr_item(t, 0)],        # is_cc
                          cls.get_len(t),
                          Position((t.pos.x, t.pos.y, t.pos.z)),
                          t.t,
                          # This is a nasty bug. It is not completely clear
                          # if this is supposed to be PDG or Geant convention.
                          # might be, that for CC neutrino events,
                          # the two vector elements might follow _different_
                          # conventions. Yep, 2 conventions for
                          # 2 vector elements...
                          #
                          # geant2pdg(t.type))
                          t.type,
                          )
                    for t in tracks], event_id)

    @classmethod
    def from_arrays(cls,
                    bjorkenys,
                    directions_x,
                    directions_y,
                    directions_z,
                    energies,
                    ids,
                    interaction_channels,
                    is_ccs,
                    lengths,
                    positions_x,
                    positions_y,
                    positions_z,
                    times,
                    types,
                    event_id,
                    ):
        directions = np.column_stack((directions_x, directions_y,
                                      directions_z))
        positions = np.column_stack((positions_x, positions_y,
                                     positions_z))
        args = bjorkenys, directions, energies, \
            ids, interaction_channels, is_ccs, lengths, positions, \
            times, types
        tracks = cls([Track(*track_args) for track_args in zip(*args)],
                     event_id)
        tracks._bjorkeny = bjorkenys
        tracks._dir = zip(directions_x, directions_y, directions_z)
        tracks._energy = energies
        tracks._id = ids
        tracks._interaction_channel = interaction_channels
        tracks._is_cc = is_ccs
        tracks._length = lengths
        tracks._pos = zip(positions_x, positions_y, positions_z)
        tracks._time = times
        tracks._type = types
        return tracks

    @classmethod
    def from_km3df(cls, km3df):
        return cls.from_table(km3df.conv_to(to='numpy'),
                              event_id=int(km3df['event_id']))

    @classmethod
    def from_table(cls, table, event_id):
        return cls([Track(
            row['bjorkeny'],
            np.array((row['dir_x'], row['dir_y'], row['dir_z'],)),
            row['energy'],
            row['id'],
            row['interaction_channel'],
            row['is_cc'],
            row['length'],
            np.array((row['pos_x'], row['pos_y'], row['pos_z'],)),
            row['time'],
            row['type']
        ) for row in table], event_id)

    @classmethod
    def deserialise(cls, *args, **kwargs):
        return cls.conv_from(*args, **kwargs)

    def serialise(self, *args, **kwargs):
        return self.conv_to(*args, **kwargs)

    @classmethod
    def conv_from(cls, data, event_id, fmt='numpy', h5loc='/'):
        if fmt == 'numpy':
            return cls.from_table(data, event_id)

    def conv_to(self, to='numpy'):
        if to == 'numpy':
            return KM3Array(np.array(self.__array__(), dtype=self.dtype),
                            h5loc=self.h5loc)
        if to == 'pandas':
            return KM3DataFrame(self.conv_to(to='numpy'), h5loc=self.h5loc)

    def __array__(self):
        return [(
            t.bjorkeny, t.dir[0], t.dir[1], t.dir[2], t.energy,
            t.id, t.interaction_channel, t.is_cc,
            t.length, t.pos[0], t.pos[1], t.pos[2], t.time, t.type,
            self.event_id,
        ) for t in self._tracks]

    @classmethod
    def get_len(cls, track):
        try:
            return track.len
        except AttributeError:
            return 0

    @classmethod
    def get_usr_item(cls, track, index):
        try:
            item = track.usr[index]
        except IndexError:
            item = 0.
        return item

    @property
    def highest_energetic_muon(self):
        if self._highest_energetic_muon is None:
            muons = [track for track in self if abs(track.type) == 13]
            if len(muons) == 0:
                raise AttributeError("No muon found")
            self._highest_energetic_muon = max(muons, key=lambda m: m.energy)
        return self._highest_energetic_muon

    def __iter__(self):
        return self

    def __iter__(self):
        return self

    @property
    def bjorkeny(self):
        if self._bjorkeny is None:
            self._bjorkeny = np.array([t.bjorkeny for t in self._tracks])
        return self._bjorkeny

    @property
    def is_cc(self):
        if self._is_cc is None:
            self._is_cc = np.array([t.is_cc for t in self._tracks])
        return self._is_cc

    @property
    def interaction_channel(self):
        if self._interaction_channel is None:
            self._interaction_channel = np.array([t.interaction_channel for
                                                  t in self._tracks])
        return self._interaction_channel

    @property
    def id(self):
        if self._id is None:
            self._id = np.array([t.id for t in self._tracks])
        return self._id

    @property
    def time(self):
        if self._time is None:
            self._time = np.array([t.time for t in self._tracks])
        return self._time

    @property
    def energy(self):
        if self._energy is None:
            self._energy = np.array([t.energy for t in self._tracks])
        return self._energy

    @property
    def length(self):
        if self._length is None:
            self._length = np.array([t.length for t in self._tracks])
        return self._length

    @property
    def type(self):
        if self._type is None:
            self._type = np.array([t.type for t in self._tracks])
        return self._type

    @property
    def pos(self):
        if self._pos is None:
            self._pos = np.array([t.pos for t in self._tracks])
        return self._pos

    @property
    def dir(self):
        if self._dir is None:
            self._dir = np.array([t.dir for t in self._tracks])
        return self._dir

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        if self._index >= len(self):
            self._index = 0
            raise StopIteration
        item = self._tracks[self._index]
        self._index += 1
        return item

    def __len__(self):
        return len(self._tracks)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._tracks[index]
        elif isinstance(index, slice):
            return self._slice_generator(index)
        else:
            raise TypeError("index must be int or slice")

    def _slice_generator(self, index):
        """A simple slice generator for iterations"""
        start, stop, step = index.indices(len(self))
        for i in range(start, stop, step):
            yield self._tracks[i]

    def __str__(self):
        n_tracks = len(self)
        plural = 's' if n_tracks > 1 or n_tracks == 0 else ''
        return("TrackSeries with {0} track{1}.".format(len(self), plural))

    def __repr__(self):
        return self.__str__()

    def __insp__(self):
        return '\n'.join([str(track) for track in self._tracks])


class SummaryframeSeries(object):
    """Collection of summary frames.
    """
    dtype = np.dtype([
        ('dom_id', '<u4'),
        ('max_sequence_number', '<u4'),
        ('n_received_packets', '<u4'),
        ('slice_id', '<u4'),
        ])
    h5loc = '/'

    def __init__(self, arr):
        self._arr = arr
        self._index = 0
        self._frames = None

    @classmethod
    def from_arrays(cls, dom_ids, max_sequence_numbers, n_received_packets,
                    slice_id):
        length = dom_ids.shape[0]
        frames = np.empty(length, cls.dtype)
        frames['dom_id'] = dom_ids
        frames['max_sequence_number'] = max_sequence_numbers
        frames['n_received_packets'] = n_received_packets
        frames['slice_id'] = np.full(length, slice_id, dtype='<u4')
        return cls(frames)

    def from_table(cls, table, slice_id):
        if slice_id is None:
            slice_id = table[0]['slice_id']
        return cls(np.array([(
            row['dom_id'],
            row['max_sequence_number'],
            row['n_received_packets'],
        ) for row in table], dtype=cls.dtype), slice_id)

    @classmethod
    def deserialise(cls, *args, **kwargs):
        return cls.conv_from(*args, **kwargs)

    def serialise(self, *args, **kwargs):
        return self.conv_to(*args, **kwargs)

    @classmethod
    def conv_from(cls, data, slice_id, fmt='numpy', h5loc='/'):
        if fmt == 'numpy':
            # return cls.from_table(data, event_id)
            return cls(data)

    def conv_to(self, to='numpy'):
        if to == 'numpy':
            return KM3Array(np.array(self.__array__(), dtype=self.dtype),
                            h5loc=self.h5loc)
        if to == 'pandas':
            return KM3DataFrame(self.conv_to(to='numpy'), h5loc=self.h5loc)

    @property
    def n_received_packets(self):
        return self._arr['n_received_packets']

    @property
    def dom_ids(self):
        return self._arr['dom_ids']

    @property
    def max_sequence_numbers(self):
        return self._arr['max_sequence_numbers']

    def __array__(self):
        return self._arr

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        if self._index >= len(self):
            self._index = 0
            raise StopIteration
        data = self._arr[self._index]
        self._index += 1
        return data

    def __len__(self):
        return self._arr.shape[0]

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._arr[index]
        elif isinstance(index, slice):
            return self._slice_generator(index)
        else:
            raise TypeError("index must be int or slice")

    def _slice_generator(self, index):
        """A simple slice generator for iterations"""
        start, stop, step = index.indices(len(self))
        for i in range(start, stop, step):
            yield self._arr[i]


class KM3Array(np.ndarray):
    """Numpy NDarray + metadata.

    This class adds the following to ``np.ndarray``:

    Attributes
    ----------
    h5loc: str, default='/'
        HDF5 group where to write into.

    Methods
    -------
    deserialise(data, h5loc='/', fmt='pandas')
        Factory to init from data.
    serialise(to='numpy')
        Convert for storage.
    """
    def __new__(cls, arr, h5loc='/'):
        obj = np.asarray(arr).view(cls)
        obj.h5loc = h5loc
        return obj

    def __init__(self, arr, h5loc='/'):
        self.array = arr
        self.h5loc = h5loc

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.h5loc = getattr(obj, 'h5loc', None)

    @classmethod
    def deserialise(cls, *args, **kwargs):
        return cls.conv_from(*args, **kwargs)

    def serialise(self, *args, **kwargs):
        return self.conv_to(*args, **kwargs)

    @classmethod
    def conv_from(cls, data, event_id=None, h5loc='/', fmt='numpy', **kwargs):
        if fmt == 'numpy':
            arr = cls(data, h5loc, **kwargs)
            return arr

    def conv_to(self, to='numpy', **kwargs):
        if to == 'numpy':
            return self
        if to == 'pandas':
            return KM3DataFrame.conv_from(self, fmt='numpy', **kwargs)

    @classmethod
    def from_dict(cls, map, dtype=None, **kwargs):
        if dtype is None:
            dtype = np.dtype([(key, float) for key in sorted(map.keys())])
        return cls(np.array([tuple((map[key] for key in dtype.names))],
                            dtype=dtype), **kwargs)


class KM3DataFrame(pd.DataFrame):
    """Pandas Dataframe + metadata.

    This class adds the following to ``pd.DataFrame``:

    Attributes
    ----------
    h5loc: str, default='/'
        HDF5 group where to write into.

    Methods
    -------
    deserialise(data, h5loc='/', fmt='pandas')
        Factory to init from data.
    serialise(to='numpy')
        Convert for storage.
    """

    # do not rename this!
    # http://pandas.pydata.org/pandas-docs/stable/internals.html#define-original-properties
    # this is preserved over df manipulations
    _metadata = ['h5loc']

    # default value
    h5loc = '/'

    @property
    def _constructor(self):
        return KM3DataFrame

    def __init__(self, *args, **kwargs):
        h5loc = kwargs.pop('h5loc', '/')
        super(KM3DataFrame, self).__init__(*args, **kwargs)
        self.h5loc = h5loc

    @classmethod
    def deserialise(cls, *args, **kwargs):
        return cls.conv_from(*args, **kwargs)

    def serialise(self, *args, **kwargs):
        return self.conv_to(*args, **kwargs)

    @classmethod
    def conv_from(cls, data, event_id=None, h5loc='/', fmt='pandas', **kwargs):
        if fmt in {'numpy', 'pandas'}:
            return cls(data, h5loc=h5loc, **kwargs)
        if fmt == 'dict':
            return cls(data, index=[0], h5loc=h5loc, **kwargs)

    def conv_to(self, to='numpy', **kwargs):
        if to == 'numpy':
            return KM3Array(self.to_records(index=False), h5loc=self.h5loc,
                            **kwargs)
        if to == 'pandas':
            return self


deserialise_map = {
    'McHits': HitSeries,
    'Hits': HitSeries,
    'TimesliceHits': TimesliceHitSeries,
    'McTracks': TrackSeries,
    'EventInfo': EventInfo,
    'SummarysliceInfo': SummarysliceInfo,
    'SummaryframeInfo': SummaryframeInfo,
    'Tracks': TrackSeries,
}
