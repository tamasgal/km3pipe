# coding=utf-8
# Filename: test_dataclasses.py
# pylint: disable=C0111,R0904,C0103
"""
...

"""
from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose)
import pytest

from km3pipe.testing import TestCase, skip   # noqa
from km3pipe.dataclasses import (
    Table, inflate_dtype, is_structured
)

__author__ = "Tamas Gal, Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal, Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestTable(TestCase):
    def setUp(self):
        self.dt = np.dtype([('a', int), ('b', float), ('group_id', int)])
        self.arr = np.array([
            (0, 1.0, 2),
            (3, 7.0, 5),
            (6, 4.0, 8),
        ], dtype=self.dt)

    def test_h5loc(self):
        tab = self.arr.view(Table)
        assert tab.h5loc == '/misc'
        tab = Table(self.arr)
        assert tab.h5loc == '/misc'
        tab = Table(self.arr, h5loc='/foo')
        assert tab.h5loc is '/foo'

    def test_view(self):
        tab = self.arr.view(Table)
        assert tab.dtype == self.dt
        assert tab.h5loc == '/misc'
        assert_array_equal(tab.a, np.array([0, 3, 6]))
        assert tab[0]['group_id'] == 2
        assert tab[0].group_id == 2
        assert tab['group_id'][0] == 2
        assert tab.group_id[0] == 2
        assert isinstance(tab[0], np.record)
        for row in tab:
            assert isinstance(row, np.record)
            assert row['a'] == 0
            assert row.a == 0
            for c in row:
                assert c == 0
                break
            assert_allclose([0, 1., 2], [c for c in row])
            break

    def test_init(self):
        tab = Table(self.arr)
        assert tab.h5loc == '/misc'
        tab = Table(self.arr, h5loc='/bla')
        assert tab.dtype == self.dt
        assert tab.h5loc == '/bla'
        assert_array_equal(tab.a, np.array([0, 3, 6]))
        assert tab[0]['group_id'] == 2
        assert tab[0].group_id == 2
        assert tab['group_id'][0] == 2
        assert tab.group_id[0] == 2
        assert isinstance(tab[0], np.record)
        for row in tab:
            assert isinstance(row, np.record)
            assert row['a'] == 0
            assert row.a == 0
            for c in row:
                assert c == 0
                break
            assert_allclose([0, 1., 2], [c for c in row])
            break

    def test_fromdict(self):
        n = 5
        dmap = {
            'a': np.ones(n, dtype=int),
            'b': np.zeros(n, dtype=float),
            'c': 0,
        }
        # tab = Table.from_dict(dmap)
        # self.assertTrue(isinstance(tab, Table))
        # assert tab.h5loc == '/misc'
        dt = [('a', float), ('b', float), ('c', float)]
        tab = Table.from_dict(dmap, dtype=dt)
        assert tab.h5loc == '/misc'
        assert isinstance(tab, Table)
        tab = Table.from_dict(dmap, dtype=dt, h5loc='/foo')
        assert tab.h5loc == '/foo'
        assert isinstance(tab, Table)
        bad_dt = [('a', float), ('b', float), ('c', float), ('d', int)]
        with pytest.raises(KeyError):
            tab = Table.from_dict(dmap, dtype=bad_dt)

    def test_fromdict_init(self):
        n = 5
        dmap = {
            'a': np.ones(n, dtype=int),
            'b': np.zeros(n, dtype=float),
            'c': 0,
        }
        dt = [('a', float), ('b', float), ('c', float)]
        tab = Table(dmap, dtype=dt)
        assert tab.h5loc == '/misc'
        self.assertTrue(isinstance(tab, Table))
        tab = Table(dmap, dtype=dt, h5loc='/foo')
        assert tab.h5loc == '/foo'
        self.assertTrue(isinstance(tab, Table))
        bad_dt = [('a', float), ('b', float), ('c', float), ('d', int)]
        with pytest.raises(KeyError):
            tab = Table(dmap, dtype=bad_dt)

    def test_append_columns(self):
        tab = Table(self.arr)
        tab = tab.append_columns('new', [1, 2, 3, 4])
        self.assertEqual(1, tab.new[0])

    def test_template(self):
        n = 10
        channel_ids = np.arange(n)
        dom_ids = np.arange(n)
        times = np.arange(n)
        tots = np.arange(n)
        triggereds = np.ones(n)
        d_hits = {
            'channel_id': channel_ids,
            'dom_id': dom_ids,
            'time': times,
            'tot': tots,
            'triggered': triggereds,
            'group_id': 0,      # event_id
        }
        tab = Table.from_template(d_hits, 'RawHitSeries')
        assert isinstance(tab, Table)
        ar_hits = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=float)
        tab = Table.from_template(ar_hits, 'RawHitSeries')
        assert isinstance(tab, Table)

    def test_sort(self):
        dt = np.dtype([('a', int), ('b', float), ('c', int)])
        arr = np.array([
            (0, 1.0, 2),
            (3, 7.0, 5),
            (6, 4.0, 8),
        ], dtype=dt)
        tab = Table(arr)
        tab_sort = tab.sorted('b')
        assert_array_equal(tab_sort['a'], np.array([0, 6, 3]))

    def test_from_2d(self):
        l2d = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        names = ['a', 'origin', 'pmt_id', 'time', 'group_id']
        dta = inflate_dtype(l2d, names)

class TestDtypes(TestCase):
    def setUp(self):
        self.c_dt = np.dtype([('a', '<f4'), ('origin', '<u4'),
                              ('pmt_id', '<u4'), ('time', '<f8'),
                              ('group_id', '<u4')])

    def test_is_structured(self):
        assert is_structured(np.ones(2, dtype=self.c_dt))
        assert not is_structured(np.ones(2, dtype=float))
        assert not is_structured(np.ones(2, dtype=int))
        assert not is_structured([1, 2, 3])
        assert not is_structured([1.0, 2.0, 3.0])
        assert not is_structured([1.0, 2, 3.0])

    def test_inflate(self):
        arr = np.ones(3, dtype=self.c_dt)
        names = ['a', 'b', 'c']
        print(arr.dtype)
        assert is_structured(arr)
        dt_a = inflate_dtype(arr, names=names)
        assert dt_a == self.c_dt
        assert not is_structured([1, 2, 3])
        dt_l = inflate_dtype([1, 2, 3], names=names)
        assert dt_l == np.dtype([('a', '<i8'), ('b', '<i8'), ('c', '<i8')])
