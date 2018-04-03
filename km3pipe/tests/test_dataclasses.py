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
    Table
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
            (3, 4.0, 5),
            (6, 7.0, 8),
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
        self.assertTrue(isinstance(tab, Table))
        tab = Table.from_dict(dmap, dtype=dt, h5loc='/foo')
        assert tab.h5loc == '/foo'
        self.assertTrue(isinstance(tab, Table))
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

    def test_append_fields(self):
        tab = Table(self.arr)
        tab = tab.append_fields('new', [1, 2, 3, 4])
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
