# coding=utf-8
# Filename: dataclasses.py
# pylint: disable=W0232,C0103,C0111
"""
...

"""
from __future__ import division, absolute_import, print_function

import numpy as np

from km3pipe.tools import angle_between

__all__ = ('Point', 'Position', 'Direction')


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


class Position(Point):
    """Represents a point in a 3D space"""
    pass


class Direction(Point):
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


class HitSeries(object):
    @classmethod
    def from_hdf5(cls, dictionary):
        return cls(dictionary, like='dict')

    def __init__(self, data, like):
        self.data = data
        self.like = like
        self._index = 0
        if like == 'dict':
            self.hit_constructor = Hit.from_dict
        else:
            self.hit_constructor = lambda x: x

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        if self._index >= len(self):
            self._index = 0
            raise StopIteration
        item = self.hit_constructor(self.data[self._index])
        self._index += 1
        return item

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.hit_constructor(self.data[index])
        elif isinstance(index, slice):
            return self._slice_generator(index)
        else:
            raise TypeError("index must be int or slice")

    def _slice_generator(self, index):
        """A simple slice generator for iterations"""
        start, stop, step = index.indices(len(self))
        for i in range(start, stop, step):
            yield self.hit_constructor(self.data[i])


class Hit(object):
    def __init__(self, time=None, tot=None, channel_id=None, dom_id=None,
                 data=None):
        self.time = time
        self.tot = tot
        self.channel_id = channel_id
        self.dom_id = dom_id
        self.data = data

    @classmethod
    def from_dict(cls, data):
        return cls(data['time'], data['tot'], data['channel_id'],
                   data['dom_id'], data)

    @classmethod
    def from_aanet(cls, data):
        return cls(data['t'], data['tot'], data['channel_id'],
                   data['dom_id'], data)
