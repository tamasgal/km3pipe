# Filename: test_dataclasses.py
# pylint: disable=C0111,R0904,C0103
# vim:set ts=4 sts=4 sw=4 et:
"""
...

"""

import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose)
import pytest

from km3pipe.testing import TestCase, skip    # noqa
from km3pipe.dataclasses import (
    Table, inflate_dtype, has_structured_dt, is_structured, DEFAULT_H5LOC,
    DEFAULT_NAME, DEFAULT_SPLIT
)

__author__ = "Tamas Gal, Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal, Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestDtypes(TestCase):
    def setUp(self):
        self.c_dt = np.dtype([('a', '<f4'), ('origin', '<u4'),
                              ('pmt_id', '<u4'), ('time',
                                                  '<f8'), ('group_id', '<u4')])

    def test_is_structured(self):
        assert is_structured(self.c_dt)
        assert not is_structured(np.dtype('int64'))
        assert not is_structured(np.dtype(int))
        assert not is_structured(np.dtype(float))

    def test_has_structured_dt(self):
        assert has_structured_dt(np.ones(2, dtype=self.c_dt))
        assert not has_structured_dt(np.ones(2, dtype=float))
        assert not has_structured_dt(np.ones(2, dtype=int))
        assert not has_structured_dt([1, 2, 3])
        assert not has_structured_dt([1.0, 2.0, 3.0])
        assert not has_structured_dt([1.0, 2, 3.0])
        assert not has_structured_dt([])

    def test_inflate_hasstructured(self):
        arr = np.ones(3, dtype=self.c_dt)
        names = ['a', 'b', 'c']
        print(arr.dtype)
        assert has_structured_dt(arr)
        dt_a = inflate_dtype(arr, names=names)
        assert dt_a == self.c_dt

    def test_inflate_nostructured(self):
        names = ['a', 'b', 'c']
        arr = [1, 2, 3]
        assert not has_structured_dt(arr)
        dt_l = inflate_dtype(arr, names=names)
        assert dt_l == np.dtype([('a', '<i8'), ('b', '<i8'), ('c', '<i8')])

    def test_inflate_mixed_casts_up(self):
        arr = [1, 2, 3.0]
        names = ['a', 'b', 'c']
        assert not has_structured_dt(arr)
        dt_a = inflate_dtype(arr, names=names)
        assert dt_a == np.dtype([('a', '<f8'), ('b', '<f8'), ('c', '<f8')])


class TestTable(TestCase):
    def setUp(self):
        self.dt = np.dtype([('a', int), ('b', float), ('group_id', int)])
        self.arr = np.array([
            (0, 1.0, 2),
            (3, 7.0, 5),
            (6, 4.0, 8),
        ],
                            dtype=self.dt)

    def test_h5loc(self):
        tab = self.arr.view(Table)
        assert tab.h5loc == DEFAULT_H5LOC
        tab = Table(self.arr)
        assert tab.h5loc == DEFAULT_H5LOC
        tab = Table(self.arr, h5loc='/foo')
        assert tab.h5loc == '/foo'

    def test_split(self):
        tab = self.arr.view(Table)
        assert tab.split_h5 is False
        tab = Table(self.arr)
        assert tab.split_h5 is False
        tab = Table(self.arr, split_h5=True)
        assert tab.split_h5

    def test_name(self):
        tab = self.arr.view(Table)
        assert tab.name == DEFAULT_NAME
        tab = Table(self.arr)
        assert tab.name == DEFAULT_NAME
        tab = Table(self.arr, name='foo')
        assert tab.name == 'foo'

    def test_view(self):
        tab = self.arr.view(Table)
        assert tab.dtype == self.dt
        assert tab.h5loc == DEFAULT_H5LOC
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
        assert tab.h5loc == DEFAULT_H5LOC
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
        # assert tab.h5loc == DEFAULT_H5LOC
        dt = [('a', float), ('b', float), ('c', float)]
        tab = Table.from_dict(dmap, dtype=dt)
        assert tab.h5loc == DEFAULT_H5LOC
        assert isinstance(tab, Table)
        tab = Table.from_dict(dmap, dtype=dt, h5loc='/foo')
        assert tab.h5loc == '/foo'
        assert isinstance(tab, Table)
        bad_dt = [('a', float), ('b', float), ('c', float), ('d', int)]
        with pytest.raises(KeyError):
            tab = Table.from_dict(dmap, dtype=bad_dt)

    def test_from_dict_without_dtype(self):
        data = {'b': [1, 2], 'c': [3, 4], 'a': [5, 6]}
        tab = Table.from_dict(data)
        assert np.allclose([1, 2], tab.b)
        assert np.allclose([3, 4], tab.c)
        assert np.allclose([5, 6], tab.a)

    def test_from_dict_with_unordered_columns_wrt_to_dtype_fields(self):
        data = {'b': [1, 2], 'c': [3, 4], 'a': [5, 6]}
        dt = [('a', float), ('b', float), ('c', float)]
        tab = Table.from_dict(data, dtype=dt)
        assert np.allclose([1, 2], tab.b)
        assert np.allclose([3, 4], tab.c)
        assert np.allclose([5, 6], tab.a)

    def test_fromcolumns(self):
        n = 5
        dlist = [
            np.ones(n, dtype=int),
            np.zeros(n, dtype=float),
            0,
        ]
        dt = np.dtype([('a', float), ('b', float), ('c', float)])
        with pytest.raises(ValueError):
            tab = Table(dlist, dtype=dt)
        tab = Table.from_columns(dlist, dtype=dt)
        print(tab.dtype)
        print(tab.shape)
        print(tab)
        assert tab.h5loc == DEFAULT_H5LOC
        assert isinstance(tab, Table)
        tab = Table.from_columns(dlist, dtype=dt, h5loc='/foo')
        print(tab.dtype)
        print(tab.shape)
        print(tab)
        assert tab.h5loc == '/foo'
        assert isinstance(tab, Table)
        bad_dt = [('a', float), ('b', float), ('c', float), ('d', int)]
        with pytest.raises(ValueError):
            tab = Table.from_columns(dlist, dtype=bad_dt)
            print(tab.dtype)
            print(tab.shape)
            print(tab)

    def test_from_columns_with_colnames(self):
        t = Table.from_columns([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
                                [13, 14, 15], [16, 17, 18], [19, 20, 21]],
                               colnames=['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        print("t.a: {}".format(t.a))
        assert np.allclose([1, 2, 3], t.a)
        print("t.b: {}".format(t.b))
        assert np.allclose([4, 5, 6], t.b)

    def test_from_columns_with_colnames_upcasts(self):
        t = Table.from_columns([[1, 2, 3], [4, 5.0, 6]], colnames=['a', 'b'])
        assert t.dtype == np.dtype([('a', float), ('b', float)])

    def test_from_columns_with_mismatching_columns_and_dtypes_raises(self):
        with pytest.raises(ValueError):
            Table.from_columns([[1, 2, 3], [4, 5, 6]],
                               dtype=np.dtype([('a', 'f4')]))

    def test_from_rows_with_colnames(self):
        t = Table.from_rows([[1, 2], [3, 4], [5, 6]], colnames=['a', 'b'])
        assert t.dtype == np.dtype([('a', int), ('b', int)])
        assert np.allclose([1, 3, 5], t.a)
        assert np.allclose([2, 4, 6], t.b)

    def test_from_rows_with_colnames_upcasts(self):
        t = Table.from_rows([[1, 2], [3.0, 4], [5, 6]], colnames=['a', 'b'])
        assert t.dtype == np.dtype([('a', float), ('b', float)])

    def test_from_rows_dim(self):
        t = Table.from_rows([[1, 2], [3.0, 4], [5, 6]], colnames=['a', 'b'])
        assert t.shape == (3, )

    def test_from_columns_dim(self):
        t = Table.from_columns([[1, 2, 3], [4, 5.0, 6]], colnames=['a', 'b'])
        assert t.shape == (3, )

    def test_fromrows(self):
        dlist = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        dt = np.dtype([('a', float), ('b', float), ('c', float)])
        with pytest.raises(ValueError):
            tab = Table(dlist, dtype=dt)
        tab = Table.from_rows(dlist, dtype=dt)
        print(tab.dtype)
        print(tab.shape)
        print(tab)
        assert tab.h5loc == DEFAULT_H5LOC
        assert isinstance(tab, Table)
        tab = Table.from_rows(dlist, dtype=dt, h5loc='/foo')
        print(tab.dtype)
        print(tab.shape)
        print(tab)
        assert tab.h5loc == '/foo'
        assert isinstance(tab, Table)
        bad_dt = [('a', float), ('b', float), ('c', float), ('d', int)]
        with pytest.raises(ValueError):
            tab = Table.from_rows(dlist, dtype=bad_dt)
            print(tab.dtype)
            print(tab.shape)
            print(tab)

    def test_expand_scalars(self):
        dmap = {
            'a': 1,
            'b': 0.,
            'c': 0,
        }
        t1 = Table._expand_scalars(dmap)
        assert len(t1) > 0
        dmap2 = {
            'a': [1, 2, 1],
            'b': 0.,
            'c': [0, 1],
        }
        t2 = Table._expand_scalars(dmap2)
        assert len(t2) > 0
        dmap3 = {
            'a': [1, 2, 1],
            'b': [0.],
            'c': [0, 1],
        }
        t3 = Table._expand_scalars(dmap3)
        assert len(t3) > 0
        dmap4 = {
            'a': [1, 2, 1],
            'b': np.array(0.),
            'c': [0, 1],
        }
        t4 = Table._expand_scalars(dmap4)
        assert len(t4) > 0
        dmap5 = {
            'a': [1, 2, 1],
            'b': np.array([1]),
            'c': [0, 1],
        }
        t5 = Table._expand_scalars(dmap5)
        assert len(t5) > 0

    def test_from_flat_dict(self):
        dmap = {
            'a': 1,
            'b': 0.,
            'c': 0,
        }
        # tab = Table.from_dict(dmap)
        # self.assertTrue(isinstance(tab, Table))
        # assert tab.h5loc == DEFAULT_H5LOC
        dt = [('a', float), ('b', float), ('c', float)]
        tab = Table.from_dict(dmap, dtype=dt)
        assert tab.h5loc == DEFAULT_H5LOC
        assert isinstance(tab, Table)
        tab = Table.from_dict(dmap, dtype=dt, h5loc='/foo')
        assert tab.h5loc == '/foo'
        assert isinstance(tab, Table)
        bad_dt = [('a', float), ('b', float), ('c', float), ('d', int)]
        with pytest.raises(KeyError):
            tab = Table.from_dict(dmap, dtype=bad_dt)

    def test_from_mixed_dict(self):
        dmap = {
            'a': 1,
            'b': 0.,
            'c': np.zeros(4),
        }
        # tab = Table.from_dict(dmap)
        # self.assertTrue(isinstance(tab, Table))
        # assert tab.h5loc == DEFAULT_H5LOC
        dt = [('a', float), ('b', float), ('c', float)]
        tab = Table.from_dict(dmap, dtype=dt)
        assert tab.h5loc == DEFAULT_H5LOC
        assert isinstance(tab, Table)
        tab = Table.from_dict(dmap, dtype=dt, h5loc='/foo')
        assert tab.h5loc == '/foo'
        assert isinstance(tab, Table)
        bad_dt = [('a', float), ('b', float), ('c', float), ('d', int)]
        with pytest.raises(KeyError):
            tab = Table.from_dict(dmap, dtype=bad_dt)

    def test_from_2d(self):
        l2d = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
        names = ['a', 'origin', 'pmt_id', 'time', 'group_id']
        dta = inflate_dtype(l2d, names)
        with pytest.raises(ValueError):
            t = Table(l2d)
        with pytest.raises(ValueError):
            t = Table(l2d, dtype=None)
        with pytest.raises(ValueError):
            t = Table(l2d, colnames=names)
        with pytest.raises(ValueError):
            t = Table(l2d, dtype=dta)
        with pytest.raises(ValueError):
            t = Table(l2d, dtype=dta, colnames=['a', 'b', 'c', 'd'])    # noqa

    def test_flat_raises(self):
        with pytest.raises(ValueError):
            t = Table([1, 2, 3], dtype=int).dtype
        with pytest.raises(ValueError):
            t = Table([1, 2, 3], dtype=float).dtype
        with pytest.raises(ValueError):
            t = Table([1, 2, 3], dtype=None).dtype
        with pytest.raises(ValueError):
            t = Table([1, 2, 3]).dtype
        with pytest.raises(ValueError):
            t = Table([1, 2, 3], colnames=['a', 'b', 'c'])    # noqa

    def test_init_with_unstructured_raises_valueerror(self):
        with pytest.raises(ValueError):
            Table(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_fromdict_init(self):
        n = 5
        dmap = {
            'a': np.ones(n, dtype=int),
            'b': np.zeros(n, dtype=float),
            'c': 0,
        }
        dt = [('a', float), ('b', float), ('c', float)]
        tab = Table(dmap, dtype=dt)
        assert tab.h5loc == DEFAULT_H5LOC
        self.assertTrue(isinstance(tab, Table))
        tab = Table(dmap, dtype=dt, h5loc='/foo')
        assert tab.h5loc == '/foo'
        self.assertTrue(isinstance(tab, Table))
        bad_dt = [('a', float), ('b', float), ('c', float), ('d', int)]
        with pytest.raises(KeyError):
            tab = Table(dmap, dtype=bad_dt)

    def test_append_columns(self):
        tab = Table(self.arr)
        print(tab)
        with pytest.raises(ValueError):
            tab = tab.append_columns('new', [1, 2, 3, 4])
        tab = tab.append_columns('new', [1, 2, 3])
        print(tab)
        assert tab.new[0] == 1
        assert tab.new[-1] == 3
        tab = tab.append_columns('bar', 0)
        print(tab)
        assert tab.bar[0] == 0
        assert tab.bar[-1] == 0
        tab = tab.append_columns('lala', [1])
        print(tab)
        assert tab.lala[0] == 1
        assert tab.lala[-1] == 1
        with pytest.raises(ValueError):
            tab = tab.append_columns(['m', 'n'], [1, 2])
        with pytest.raises(ValueError):
            tab = tab.append_columns(['m', 'n'], [[1], [2]])
        tab = tab.append_columns(['m', 'n'], [[1, 1, 2], [2, 4, 5]])
        print(tab)
        assert tab.m[0] == 1
        assert tab.m[-1] == 2
        assert tab.n[0] == 2
        assert tab.n[-1] == 5

    def test_append__single_column(self):
        tab = Table({'a': 1})
        print(tab.dtype)
        tab = tab.append_columns(['b'], np.array([[2]]))
        print(tab.dtype)
        print(tab.b)

    def test_append_columns_with_single_value(self):
        tab = Table({'a': 1})
        tab = tab.append_columns('group_id', 0)
        assert 0 == tab.group_id[0]

    def test_append_columns_with_multiple_values(self):
        tab = Table({'a': [1, 2]})
        tab = tab.append_columns('group_id', [0, 1])
        assert 0 == tab.group_id[0]
        assert 1 == tab.group_id[1]

    def test_append_columns_modifies_dtype(self):
        tab = Table({'a': [1, 2]})
        tab = tab.append_columns('group_id', [0, 1])
        assert 'group_id' in tab.dtype.names

    def test_append_column_which_is_too_short_raises(self):
        tab = Table({'a': [1, 2, 3]})
        with pytest.raises(ValueError):
            tab = tab.append_columns('b', [4, 5])

    def test_append_columns_duplicate(self):
        tab = Table({'a': 1})
        with pytest.raises(ValueError):
            tab = tab.append_columns(['a'], np.array([[2]]))

    def test_append_columns_with_mismatching_lengths_raises(self):
        tab = Table({'a': [1, 2, 3]})
        with pytest.raises(ValueError):
            tab.append_columns(colnames=['b', 'c'], values=[[4, 5, 6], [7, 8]])

    def test_append_columns_which_is_too_long(self):
        tab = Table({'a': [1, 2, 3]})
        with pytest.raises(ValueError):
            tab.append_columns('b', values=[4, 5, 6, 7])

    def test_drop_column(self):
        tab = Table({'a': 1, 'b': 2})
        tab = tab.drop_columns('a')
        with pytest.raises(AttributeError):
            print(tab.a)
        print(tab.b)

    def test_drop_columns(self):
        tab = Table({'a': 1, 'b': 2, 'c': 3})
        print(tab.dtype)
        tab = tab.drop_columns(['a', 'b'])
        print(tab.dtype)
        with pytest.raises(AttributeError):
            print(tab.a)
        with pytest.raises(AttributeError):
            print(tab.b)
        print(tab.c)

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
            'group_id': 0,    # event_id
        }
        tab = Table.from_template(d_hits, 'Hits')
        assert tab.name == 'Hits'
        assert tab.split_h5 is True
        assert isinstance(tab, Table)
        ar_hits = {
            'channel_id': np.ones(n, dtype=int),
            'dom_id': np.ones(n, dtype=int),
            'time': np.ones(n, dtype=float),
            'tot': np.ones(n, dtype=float),
            'triggered': np.ones(n, dtype=bool),
            'group_id': np.ones(n, dtype=int),
        }
        tab = Table.from_template(ar_hits, 'Hits')
        assert tab.name == 'Hits'
        assert tab.split_h5 is True
        assert isinstance(tab, Table)

    def test_incomplete_template(self):
        n = 10
        channel_ids = np.arange(n)
        dom_ids = np.arange(n)
        # times = np.arange(n)
        tots = np.arange(n)
        triggereds = np.ones(n)
        d_hits = {
            'channel_id': channel_ids,
            'dom_id': dom_ids,
        # 'time': times,
            'tot': tots,
            'triggered': triggereds,
            'group_id': 0,    # event_id
        }
        with pytest.raises(KeyError):
            tab = Table.from_template(d_hits, 'Hits')
            assert tab is not None
        ar_hits = {
            'channel_id': np.ones(n, dtype=int),
            'dom_id': np.ones(n, dtype=int),
        # 'time': np.ones(n, dtype=float),
            'tot': np.ones(n, dtype=float),
            'triggered': np.ones(n, dtype=bool),
            'group_id': np.ones(n, dtype=int),
        }
        with pytest.raises(KeyError):
            tab = Table.from_template(ar_hits, 'Hits')
            assert tab is not None

    def test_adhoc_template(self):
        a_template = {
            'dtype': np.dtype([('a', '<u4'), ('b', 'f4')]),
            'h5loc': '/yat',
            'split_h5': True,
            'h5singleton': True,
            'name': "YetAnotherTemplate",
        }
        arr = np.array([(1, 3), (2, 4)], dtype=a_template['dtype'])
        tab = Table.from_template(arr, a_template)
        self.assertListEqual([1, 2], list(tab.a))
        self.assertListEqual([3.0, 4.0], list(tab.b))
        assert "YetAnotherTemplate" == tab.name
        assert tab.h5singleton

    def test_adhoc_noname_template(self):
        a_template = {
            'dtype': np.dtype([('a', '<u4'), ('b', 'f4')]),
            'h5loc': '/yat',
            'split_h5': True,
            'h5singleton': True,
        }
        arr = np.array([(1, 3), (2, 4)], dtype=a_template['dtype'])
        tab = Table.from_template(arr, a_template)
        self.assertListEqual([1, 2], list(tab.a))
        self.assertListEqual([3.0, 4.0], list(tab.b))
        assert DEFAULT_NAME == tab.name
        assert tab.h5singleton

    def test_element_list_with_dtype(self):
        bad_elist = [
            [1, 2.1],
            [3, 4.2],
            [5, 6.3],
        ]
        dt = np.dtype([('a', int), ('b', float)])
        print('list(list)')
        arr_bad = np.array(bad_elist, dtype=dt)
        print(arr_bad)
        elist = [tuple(el) for el in bad_elist]
        arr = np.array(elist, dtype=dt)
        print('list(tuple)')
        print(arr)
        tab = Table(arr)
        print(tab)
        assert tab.a[0] == 1

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

    def test_init_directly_with_df(self):
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        tab = Table(df, h5loc='/foo')
        assert np.allclose(df.a, tab.a)
        assert np.allclose(df.b, tab.b)
        assert tab.h5loc == '/foo'

    def test_df(self):
        from pandas.util.testing import assert_frame_equal
        import pandas as pd
        dt = np.dtype([('a', int), ('b', float), ('c', int)])
        arr = np.array([
            (0, 1.0, 2),
            (3, 7.0, 5),
            (6, 4.0, 8),
        ], dtype=dt)
        print(dir(Table))
        df = pd.DataFrame(arr)
        tab = Table.from_dataframe(df, h5loc='/bla')
        df2 = tab.to_dataframe()
        assert_frame_equal(df, df2)

    def test_slicing(self):
        dt = np.dtype([('a', int), ('b', float), ('c', bool)])
        arr = np.array([
            (0, 1.0, True),
            (2, 3.0, False),
            (4, 5.0, True),
        ],
                       dtype=dt)
        tab = Table(arr)
        assert 2 == len(tab[tab.c])
        assert 1 == len(tab[tab.b > 3.0])

    def test_contains(self):
        dt = np.dtype([('a', int), ('b', float), ('c', bool)])
        arr = np.array([
            (0, 1.0, True),
            (2, 3.0, False),
            (4, 5.0, True),
        ],
                       dtype=dt)
        tab = Table(arr)
        assert 'a' in tab
        assert 'b' in tab
        assert 'c' in tab
        assert 'd' not in tab

    def test_index_returns_reference(self):
        tab = Table({'a': [1, 2, 3]})
        tab[1].a = 4
        assert np.allclose(tab.a, [1, 4, 3])

    def test_index_of_attribute_returns_reference(self):
        tab = Table({'a': [1, 2, 3]})
        tab.a[1] = 4
        assert np.allclose(tab.a, [1, 4, 3])

    def test_mask_returns_copy(self):
        tab = Table({'a': [1, 2, 3]})
        tab[[True, False, True]].a = [4, 5]
        assert np.allclose(tab.a, [1, 2, 3])

    def test_mask_on_attribute_returns_reference(self):
        tab = Table({'a': [1, 2, 3]})
        tab.a[[True, False, True]] = [4, 5]
        assert np.allclose(tab.a, [4, 2, 5])

    def test_index_mask_returns_copy(self):
        tab = Table({'a': [1, 2, 3]})
        tab[[1, 2]].a = [4, 5]
        assert np.allclose(tab.a, [1, 2, 3])

    def test_index_mask_of_attribute_returns_reference(self):
        tab = Table({'a': [1, 2, 3]})
        tab.a[[1, 2]] = [4, 5]
        assert np.allclose(tab.a, [1, 4, 5])

    def test_slice_returns_reference(self):
        tab = Table({'a': [1, 2, 3]})
        tab[:2].a = [4, 5]
        assert np.allclose(tab.a, [4, 5, 3])

    def test_slice_of_attribute_returns_reference(self):
        tab = Table({'a': [1, 2, 3]})
        tab.a[:2] = [4, 5]
        assert np.allclose(tab.a, [4, 5, 3])

    def test_slice_keeps_metadata(self):
        tab = Table({
            'a': [1, 2, 3]
        },
                    h5loc='/lala',
                    split_h5=True,
                    name='bla',
                    h5singleton=False)
        assert tab[:2].h5loc == '/lala'
        assert tab[:2].name == 'bla'
        assert not tab[:2].h5singleton
        assert tab[:2].split_h5

    def test_mask_keeps_metadata(self):
        tab = Table({
            'a': [1, 2, 3]
        },
                    h5loc='/lala',
                    split_h5=True,
                    name='bla',
                    h5singleton=True)
        m = np.ones(len(tab), dtype=bool)
        assert tab[m].h5loc == '/lala'
        assert tab[m].name == 'bla'
        assert tab[m].h5singleton
        assert tab[m].split_h5

    def test_indexing_keeps_metadata(self):
        tab = Table({
            'a': [1, 2, 3]
        },
                    h5loc='/lala',
                    split_h5=True,
                    name='bla',
                    h5singleton=True)
        im = [1, 1, 0]
        assert tab[im].h5loc == '/lala'
        assert tab[im].name == 'bla'
        assert tab[im].h5singleton
        assert tab[im].split_h5

    def test_crash_repr(self):
        a = np.array('', dtype=[('a', '<U1')])
        with pytest.raises(TypeError):
            print(len(a))
        tab = Table(a)
        s = tab.__str__()
        assert s is not None
        r = tab.__repr__()
        assert r is not None

    def test_array_finalize_with_obj_none(self):
        tab = Table({'a': [1, 2, 3]})
        assert tab.__array_finalize__(None) is None

    def test_array_wrap(self):
        t = Table({'a': [1, 2, 3], 'b': [4, 5, 6]})
        wrapped = t.__array_wrap__(np.array((Table({'a': 1}))))
        assert wrapped.a[0] == 1

    def test_templates_avail(self):
        t = Table({'a': 1})
        templates = t.templates_avail
        assert templates

    def test_add_table_to_itself(self):
        tab = Table({'a': [1]})
        added_tab = tab + tab
        assert 2 == len(added_tab)


class TestTableFancyAttributes(TestCase):
    def setUp(self):
        self.arr_bare = Table({
            'a': [1, 2, 3],
            'b': [3, 4, 5],
        })
        self.arr_wpos = Table({
            'a': [1, 2, 3],
            'b': [3, 4, 5],
            'pos_x': [10, 20, 30],
            'pos_y': [40, 50, 60],
            'pos_z': [70, 80, 90],
            'dir_x': [10.0, 20.0, 30.0],
            'dir_y': [40.0, 50.0, 60.0],
            'dir_z': [70.0, 80.0, 90.0],
        })

    def test_pos_getter(self):
        tab = Table({
            'pos_x': [1, 2, 3],
            'pos_y': [4, 5, 6],
            'pos_z': [7, 8, 9]
        })
        assert np.allclose([[1, 4, 7], [2, 5, 8], [3, 6, 9]], tab.pos)

    def test_pos_getter_for_single_entry(self):
        tab = Table({
            'pos_x': [1, 2, 3],
            'pos_y': [4, 5, 6],
            'pos_z': [7, 8, 9]
        })
        assert np.allclose([[2, 5, 8]], tab.pos[1])

    def test_dir_getter(self):
        tab = Table({
            'dir_x': [1, 2, 3],
            'dir_y': [4, 5, 6],
            'dir_z': [7, 8, 9]
        })
        assert np.allclose([[1, 4, 7], [2, 5, 8], [3, 6, 9]], tab.dir)

    def test_dir_getter_for_single_entry(self):
        tab = Table({
            'dir_x': [1, 2, 3],
            'dir_y': [4, 5, 6],
            'dir_z': [7, 8, 9]
        })
        assert np.allclose([[2, 5, 8]], tab.dir[1])

    def test_dir_setter(self):
        tab = Table({
            'dir_x': [1, 0, 0],
            'dir_y': [0, 1, 0],
            'dir_z': [0, 0, 1]
        })
        new_dir = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        tab.dir = new_dir
        assert np.allclose(new_dir, tab.dir)

    def test_pos_setter(self):
        tab = Table({
            'pos_x': [1, 0, 0],
            'pos_y': [0, 1, 0],
            'pos_z': [0, 0, 1]
        })
        new_pos = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        tab.pos = new_pos
        assert np.allclose(new_pos, tab.pos)

    def test_same_shape_pos(self):
        with pytest.raises(AttributeError):
            p = self.arr_bare.pos
        p = self.arr_wpos.pos
        self.arr_wpos.pos = p
        assert p is not None
        # assert p.shape[1] == 3
        with pytest.raises(ValueError):
            self.arr_bare.dir = p

    def test_same_shape_dir(self):
        with pytest.raises(AttributeError):
            p = self.arr_bare.dir
        p = self.arr_wpos.dir
        self.arr_wpos.dir = p
        assert p is not None
        # assert p.shape[1] == 3
        a2 = self.arr_bare.copy()
        with pytest.raises(ValueError):
            self.arr_bare.dir = p

    def test_phi(self):
        tab = Table({
            'dir_x': [1, 0, 0],
            'dir_y': [0, 1, 0],
            'dir_z': [0, 0, 1]
        })
        p = tab.phi
        assert p is not None

    def test_phi(self):
        tab = Table({
            'dir_x': [1, 0, 0],
            'dir_y': [0, 1, 0],
            'dir_z': [0, 0, 1]
        })
        p = tab.theta
        assert p is not None

    def test_zen(self):
        tab = Table({
            'dir_x': [1, 0, 0],
            'dir_y': [0, 1, 0],
            'dir_z': [0, 0, 1]
        })
        p = tab.zenith
        assert p is not None

    def test_azi(self):
        tab = Table({
            'dir_x': [1, 0, 0],
            'dir_y': [0, 1, 0],
            'dir_z': [0, 0, 1]
        })
        p = tab.azimuth
        assert p is not None

    def test_pos_setter_if_pos_x_y_z_are_not_present_raises(self):
        tab = Table({'a': 1})
        with pytest.raises(ValueError):
            tab.pos = [[1], [2], [3]]

    def test_dir_setter_if_dir_x_y_z_are_not_present_raises(self):
        tab = Table({'a': 1})
        with pytest.raises(ValueError):
            tab.dir = [[1], [2], [3]]

    def test_triggered_keeps_attrs(self):
        n = 5
        channel_ids = np.arange(n)
        dom_ids = np.arange(n)
        times = np.arange(n)
        tots = np.arange(n)
        triggereds = np.array([0, 1, 1, 0, 1])
        hits = Table(
            {
                'channel_id': channel_ids,
                'dom_id': dom_ids,
                'time': times,
                'tot': tots,
                'triggered': triggereds,
                'group_id': 0,    # event_id
            },
            name='hits',
            h5loc='/foo',
            split_h5=True
        )
        triggered_hits = hits.triggered_rows
        assert len(triggered_hits) == 3
        assert triggered_hits.split_h5
        assert triggered_hits.name == 'hits'
        assert triggered_hits.h5loc == '/foo'

    def test_triggered_missing_col_raises(self):
        n = 5
        channel_ids = np.arange(n)
        dom_ids = np.arange(n)
        times = np.arange(n)
        tots = np.arange(n)
        hits = Table(
            {
                'channel_id': channel_ids,
                'dom_id': dom_ids,
                'time': times,
                'tot': tots,
                'group_id': 0,    # event_id
            },
            name='hits',
            h5loc='/foo',
            split_h5=True
        )
        with pytest.raises(KeyError):
            triggered_hits = hits.triggered_rows
            assert triggered_hits is not None
