#!/usr/bin/env python

import numpy as np
import tables as tb

from km3pipe.io import read_group
from km3pipe.io.pandas import H5Chain
from km3pipe.tools import insert_prefix_to_dtype
from km3pipe.testing import TestCase


class TestMultiTable(TestCase):
    def setUp(self):
        self.foo = np.array([
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
        ], dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8'), ])
        self.bar = np.array([
            (10.0, 20.0, 30.0),
            (40.0, 50.0, 60.0),
        ], dtype=[('aa', '<f8'), ('bb', '<f8'), ('cc', '<f8'), ])
        self.tabs = {'foo': self.foo, 'bar': self.bar}
        self.where = '/lala'
        self.h5name = './test.h5'
        self.h5file = tb.open_file(
            # create the file in memory only
            self.h5name, 'w', driver="H5FD_CORE", driver_core_backing_store=0)
        for name, tab in self.tabs.items():
            self.h5file.create_table(self.where, name=name, obj=tab,
                                     createparents=True)

    def tearDown(self):
        self.h5file.close()

    def test_name_insert(self):
        exp_foo = ('foo_a', 'foo_b', 'foo_c')
        exp_bar = ('bar_aa', 'bar_bb', 'bar_cc')
        pref_foo = insert_prefix_to_dtype(self.tabs['foo'], 'foo')
        pref_bar = insert_prefix_to_dtype(self.tabs['bar'], 'bar')
        self.assertEqual(exp_foo, pref_foo.dtype.names)
        self.assertEqual(exp_bar, pref_bar.dtype.names)

    #def test_group_read(self):
    #    tabs = read_group(self.h5file.root)
    #    exp_cols = (
    #        'bar_aa', 'bar_bb', 'bar_cc',
    #        'foo_a', 'foo_b', 'foo_c',
    #    )
    #    exp_shape = (2, 6)
    #    res_shape = tabs.shape
    #    res_cols = tuple(tabs.columns)
    #    print(exp_cols)
    #    print(res_cols)
    #    self.assertEqual(exp_shape, res_shape)
    #    self.assertEqual(exp_cols, res_cols)


class TestH5Chain(TestCase):
    def setUp(self):
        self.foo = np.array([
            (1.0, 2.0, 1.0, 1),
            (4.0, 5.0, 0.0, 1),
            (1.2, 2.2, 1.2, 2),
            (4.2, 5.2, 0.2, 2),
            (1.5, 2.5, 1.5, 3),
            (4.5, 5.5, 0.5, 3),
            (1.5, 2.5, 1.5, 4),
            (4.5, 5.5, 0.5, 4),
        ], dtype=[('a', '<f8'), ('b', '<f8'),
                  ('c', '<f8'), ('event_id', int)])
        self.bar = np.array([
            (10.5, 20.5, 30.5, 1),
            (40.5, 50.5, 60.5, 2),
            (11.5, 21.5, 31.5, 3),
            (41.5, 51.5, 61.5, 4),
        ], dtype=[('aa', '<f8'), ('bb', '<f8'),
                  ('cc', '<f8'), ('event_id', int)])
        self.yay = np.array([
            (10.0, 20.0, 30.0, 1),
            (40.0, 50.0, 60.0, 2),
            (12.0, 22.0, 32.0, 3),
            (42.0, 52.0, 62.0, 4),
        ], dtype=[('aaa', '<f8'), ('bbb', '<f8'),
                  ('ccc', '<f8'), ('event_id', int)])
        self.info = np.array([
            (0,  0),
            (0,  1),
            (0,  2),
            (0,  3),
        ], dtype=[('aaa', int), ('event_id', int)])
        self.tabs = {'foo': self.foo, 'bar': self.bar, 'yay': self.yay,
                     'event_info': self.info}
        self.where = {'foo': '/', 'bar': '/lala', 'yay': '/lala',
                      'event_info': '/'}
        self.h5name = './test.h5'
        self.h5name2 = './test2.h5'
        self.h5file = tb.open_file(self.h5name, 'a', driver="H5FD_CORE",
                                   driver_core_backing_store=0)
        self.h5file2 = tb.open_file(self.h5name2, 'a', driver="H5FD_CORE",
                                    driver_core_backing_store=0)
        for name, tab in self.tabs.items():
            self.h5file.create_table(self.where[name], name=name, obj=tab,
                                     createparents=True)
            self.h5file2.create_table(self.where[name], name=name, obj=tab,
                                      createparents=True)

    def tearDown(self):
        self.h5file.close()
        self.h5file2.close()

    def test_noargs(self):
        c = H5Chain({self.h5name: self.h5file, self.h5name2: self.h5file2})
        run = c()
        print(run['foo'])
        self.assertAlmostEqual(run['foo'].shape[0], 12)
        self.assertAlmostEqual(run['foo'].shape[1], 4)
        self.assertAlmostEqual(run['lala'].shape[0], 6)
        self.assertAlmostEqual(run['lala'].shape[1], 8)
        self.assertAlmostEqual(tuple(run['foo'].columns),
                               tuple(['a', 'b', 'c', 'event_id']))
        self.assertAlmostEqual(
            tuple(run['lala'].columns),
            tuple(['bar_aa', 'bar_bb', 'bar_cc', 'bar_event_id',
                   'yay_aaa', 'yay_bbb', 'yay_ccc', 'yay_event_id'])
        )

    def skip_n_events(self):
        c = H5Chain({self.h5name: self.h5file, self.h5name2: self.h5file2})
        run = c(2)
        print(run['foo'])
        print(run['lala'])
        print(run['foo'].shape)
        print(run['lala'].shape)
        self.assertAlmostEqual(run['foo'].shape[0], 16)
        self.assertAlmostEqual(run['foo'].shape[1], 2)
        self.assertAlmostEqual(run['lala'].shape[0], 8)
        self.assertAlmostEqual(run['lala'].shape[1], 2)


class TestH5Pump(TestCase):
    pass


class TestH5Sink(TestCase):
    def test_to_array(self):
        # check if is converted to array:
        # hitseries
        # reco
        pass
