#!/usr/bin/env python
"""Tests for HDF5 stuff"""
from collections import OrderedDict
import tempfile
from os.path import join, dirname

import numpy as np
import tables as tb

from km3pipe import Blob, Module, Pipeline, Pump, version
from km3pipe.dataclasses import Table
from km3pipe.io.hdf5 import (
    HDF5Pump, HDF5Sink, HDF5Header, convert_header_dict_to_table,
    FORMAT_VERSION
)
from km3pipe.tools import insert_prefix_to_dtype
from km3pipe.testing import TestCase

DATA_DIR = join(dirname(__file__), '../../kp-data/test_data/')


class TestMultiTable(TestCase):
    def setUp(self):
        self.foo = np.array([
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
        ],
                            dtype=[
                                ('a', '<f8'),
                                ('b', '<f8'),
                                ('c', '<f8'),
                            ])
        self.bar = np.array([
            (10.0, 20.0, 30.0),
            (40.0, 50.0, 60.0),
        ],
                            dtype=[
                                ('aa', '<f8'),
                                ('bb', '<f8'),
                                ('cc', '<f8'),
                            ])
        self.tabs = {'foo': self.foo, 'bar': self.bar}
        self.where = '/lala'
        self.fobj = tempfile.NamedTemporaryFile(delete=True)
        self.h5name = self.fobj.name
        self.h5file = tb.open_file(
        # create the file in memory only
            self.h5name,
            'w',
            driver="H5FD_CORE",
            driver_core_backing_store=0
        )
        for name, tab in self.tabs.items():
            self.h5file.create_table(
                self.where, name=name, obj=tab, createparents=True
            )

    def tearDown(self):
        self.h5file.close()
        self.fobj.close()

    def test_name_insert(self):
        exp_foo = ('foo_a', 'foo_b', 'foo_c')
        exp_bar = ('bar_aa', 'bar_bb', 'bar_cc')
        pref_foo = insert_prefix_to_dtype(self.tabs['foo'], 'foo')
        pref_bar = insert_prefix_to_dtype(self.tabs['bar'], 'bar')
        self.assertEqual(exp_foo, pref_foo.dtype.names)
        self.assertEqual(exp_bar, pref_bar.dtype.names)

    # def test_group_read(self):
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


class TestH5Pump(TestCase):
    def setUp(self):
        self.fname = join(DATA_DIR, 'numu_cc_test.h5')

    def test_init_sets_filename_if_no_keyword_arg_is_passed(self):
        p = HDF5Pump(self.fname)
        self.assertEqual(self.fname, p.filename)
        p.finish()

    def test_context(self):
        with HDF5Pump(self.fname) as h5:
            self.assertEqual(self.fname, h5.filename)
            assert h5[0] is not None
            for blob in h5:
                assert blob is not None
                break

    def test_standalone(self):
        pump = HDF5Pump(filename=self.fname)
        next(pump)
        pump.finish()

    def test_pipe(self):
        p = Pipeline()
        p.attach(HDF5Pump, filename=self.fname)
        p.drain()

    def test_event_info_is_not_empty(self):
        self.fname = join(DATA_DIR, 'test_event_info.h5')

        class Printer(Module):
            def process(self, blob):
                assert blob['EventInfo'].size != 0
                return blob

        p = Pipeline()
        p.attach(HDF5Pump, filename=self.fname)
        p.attach(Printer)
        p.drain()

    def test_event_info_has_correct_group_id(self):
        self.fname = join(DATA_DIR, 'test_event_info.h5')

        class Printer(Module):
            def configure(self):
                self.index = 0

            def process(self, blob):
                assert blob['EventInfo'][0].group_id == self.index
                self.index += 1
                return blob

        p = Pipeline()
        p.attach(HDF5Pump, filename=self.fname)
        p.attach(Printer)
        p.drain()


class TestH5Sink(TestCase):
    def setUp(self):
        self.fname = join(DATA_DIR, 'numu_cc_test.h5')
        self.fobj = tempfile.NamedTemporaryFile(delete=True)
        self.out = tb.open_file(
            self.fobj.name,
            "w",
            driver="H5FD_CORE",
            driver_core_backing_store=0
        )

    def tearDown(self):
        self.out.close()
        self.fobj.close()

    # def test_init_has_to_be_explicit(self):
    #     with self.assertRaises(TypeError):
    #         HDF5Sink(self.out)

    def test_pipe(self):
        p = Pipeline()
        p.attach(HDF5Pump, filename=self.fname)
        p.attach(HDF5Sink, h5file=self.out)
        p.drain()

    def test_scalars(self):
        out = tb.open_file(
            'foobar_scalar',
            "a",
            driver="H5FD_CORE",
            driver_core_backing_store=0
        )

        def pu(blob):
            return {'foo': 42.0}

        p = Pipeline()
        p.attach(pu)
        p.attach(HDF5Sink, h5file=out, keep_open=True)
        p.drain(2)
        node = out.root.misc
        assert node is not None
        assert node.cols is not None
        assert 'foo' in set(node.cols._v_colnames)
        out.close()

    def test_h5info(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Pump):
            def process(self, blob):
                return Blob()

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        with tb.open_file(fname, 'r') as h5file:
            assert version == h5file.root._v_attrs.km3pipe.decode()
            assert tb.__version__ == h5file.root._v_attrs.pytables.decode()
            assert FORMAT_VERSION == h5file.root._v_attrs.format_version

        fobj.close()


class TestH5SinkConsistency(TestCase):
    def test_h5_consistency_for_tables_without_group_id(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Pump):
            def configure(self):
                self.count = 0

            def process(self, blob):
                self.count += 10
                tab = Table({'a': self.count, 'b': 1}, h5loc='tab')
                return Blob({'tab': tab})

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        with tb.File(fname) as f:
            a = f.get_node("/tab")[:]['a']
            b = f.get_node("/tab")[:]['b']
            group_id = f.get_node("/tab")[:]['group_id']
        assert np.allclose([10, 20, 30, 40, 50], a)
        assert np.allclose([1, 1, 1, 1, 1], b)
        assert np.allclose([0, 1, 2, 3, 4], group_id)
        fobj.close()

    def test_h5_consistency_for_tables_without_group_id_and_multiple_keys(
            self
    ):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Pump):
            def configure(self):
                self.count = 0

            def process(self, blob):
                self.count += 10
                tab1 = Table({'a': self.count, 'b': 1}, h5loc='tab1')
                tab2 = Table({'c': self.count + 1, 'd': 2}, h5loc='tab2')
                return Blob({'tab1': tab1, 'tab2': tab2})

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        with tb.File(fname) as f:
            a = f.get_node("/tab1")[:]['a']
            b = f.get_node("/tab1")[:]['b']
            c = f.get_node("/tab2")[:]['c']
            d = f.get_node("/tab2")[:]['d']
            group_id_1 = f.get_node("/tab1")[:]['group_id']
            group_id_2 = f.get_node("/tab1")[:]['group_id']
        assert np.allclose([10, 20, 30, 40, 50], a)
        assert np.allclose([1, 1, 1, 1, 1], b)
        assert np.allclose([0, 1, 2, 3, 4], group_id_1)
        assert np.allclose([11, 21, 31, 41, 51], c)
        assert np.allclose([2, 2, 2, 2, 2], d)
        assert np.allclose([0, 1, 2, 3, 4], group_id_2)
        fobj.close()

    def test_h5_consistency_for_tables_with_custom_group_id(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Pump):
            def process(self, blob):
                tab = Table({'group_id': 2}, h5loc='tab')
                return Blob({'tab': tab})

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        with tb.File(fname) as f:
            group_id = f.get_node("/tab")[:]['group_id']

        assert np.allclose([2, 2, 2, 2, 2], group_id)

        fobj.close()

    def test_h5_singletons(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Pump):
            def process(self, blob):
                tab = Table({'a': 2}, h5loc='tab', h5singleton=True)
                return Blob({'tab': tab})

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        with tb.File(fname) as f:
            a = f.get_node("/tab")[:]['a']

        assert len(a) == 1

        fobj.close()

    def test_h5_singletons_reading(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Pump):
            def process(self, blob):
                tab = Table({'a': 2}, h5loc='tab', h5singleton=True)
                return Blob({'Tab': tab})

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        class Observer(Module):
            def process(self, blob):
                print(blob)
                assert 'Tab' in blob
                print(blob['Tab'])
                assert len(blob['Tab']) == 1
                assert blob['Tab'].a[0] == 2
                return blob

        pipe = Pipeline()
        pipe.attach(HDF5Pump, filename=fname)
        pipe.attach(Observer)
        pipe.drain()

        fobj.close()


class TestHDF5PumpConsistency(TestCase):
    def test_hdf5_readout(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Pump):
            def configure(self):
                self.count = 0

            def process(self, blob):
                self.count += 1
                tab = Table({'a': self.count * 10, 'b': 1}, h5loc='tab')
                tab2 = Table({'a': np.arange(self.count)}, h5loc='tab2')
                blob['Tab'] = tab
                blob['Tab2'] = tab2
                return blob

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        class BlobTester(Module):
            def configure(self):
                self.index = 0

            def process(self, blob):
                self.index += 1
                assert 'GroupInfo' in blob
                assert 'Tab' in blob
                print(self.index)
                print(blob['Tab'])
                print(blob['Tab']['a'])
                assert self.index - 1 == blob['GroupInfo'].group_id
                assert self.index * 10 == blob['Tab']['a']
                assert 1 == blob['Tab']['b'] == 1
                assert np.allclose(np.arange(self.index), blob['Tab2']['a'])
                return blob

        pipe = Pipeline()
        pipe.attach(HDF5Pump, filename=fname)
        pipe.attach(BlobTester)
        pipe.drain()

        fobj.close()

    def test_hdf5_readout_split_tables(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Pump):
            def configure(self):
                self.count = 0

            def process(self, blob):
                self.count += 1
                tab = Table({
                    'a': self.count * 10,
                    'b': 1
                },
                            h5loc='/tab',
                            split_h5=True)
                blob['Tab'] = tab
                return blob

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        class BlobTester(Module):
            def configure(self):
                self.index = 0

            def process(self, blob):
                self.index += 1
                assert 'GroupInfo' in blob
                assert 'Tab' in blob
                assert self.index - 1 == blob['GroupInfo'].group_id
                assert self.index * 10 == blob['Tab']['a']
                assert 1 == blob['Tab']['b'] == 1
                return blob

        pipe = Pipeline()
        pipe.attach(HDF5Pump, filename=fname)
        pipe.attach(BlobTester)
        pipe.drain()

        fobj.close()


class TestHDF5Header(TestCase):
    def setUp(self):
        # self.hdict = OrderedDict([
        # # yapf crushes the formatting, never mind...
        # # we use OrderedDict here to ensure the correct ordering
        #     ("param_a", OrderedDict([("field_a_1", "1"), ("field_a_2", "2")])),
        #     ("param_b", OrderedDict([("field_b_1", "a")])),
        #     ("param_c", OrderedDict([("field_c_1", 23)])),
        #     (
        #         "param_d",
        #         OrderedDict([("param_d_0", 1), ("param_d_1", 2),
        #                      ("param_d_2", 3)])
        #     )
        # ])
        self.hdict = {
            "param_a": {
                "field_a_1": "1",
                "field_a_2": "2"
            },
            "param_b": {
                "field_b_1": "a"
            },
            "param_c": {
                "field_c_1": 23
            },
            "param_d": {
                "param_d_0": 1,
                "param_d_1": 2,
                "param_d_2": 3
            },
            "param_e": {
                "param_e_2": 3,
                "param_e_0": 1,
                "param_e_1": 2
            },
        }

    def test_init(self):
        HDF5Header({})

    def test_header(self):
        header = HDF5Header(self.hdict)
        assert "1" == header.param_a.field_a_1
        assert "2" == header.param_a.field_a_2
        assert "a" == header.param_b.field_b_1
        assert 23 == header.param_c.field_c_1

    def test_header_with_vectors(self):
        header = HDF5Header(self.hdict)
        self.assertTupleEqual((1, 2, 3), header.param_d)

    def test_header_with_scrumbled_vectors(self):
        header = HDF5Header(self.hdict)
        self.assertTupleEqual((1, 2, 3), header.param_e)

    # def test_header_with_scalars(self):
    #     header = HDF5Header(self.hdict)
    #     assert 4 == header.param_e
    #     assert 5.6 == header.param_f
    #
    # def test_scientific_notation(self):
    #     header = HDF5Header(self.hdict)
    #     assert 7e+08 == header.param_g

    def test_header_from_table(self):
        table = convert_header_dict_to_table(self.hdict)
        header = HDF5Header.from_table(table)
        print(header)
        assert 1.0 == header.param_a.field_a_1
        assert 2.0 == header.param_a.field_a_2
        assert "a" == header.param_b.field_b_1
        assert 23 == header.param_c.field_c_1
        self.assertTupleEqual((1, 2, 3), header.param_d)

    def test_header_from_hdf5_file(self):
        header = HDF5Header.from_hdf5(join(DATA_DIR, 'raw_header.h5'))
        assert 'MUSIC' == header.propag[0]
        assert 'seawater' == header.propag[1]
        assert 3450 == header.seabottom[0]
        self.assertAlmostEqual(12.1, header.livetime.numberOfSeconds, places=3)
        self.assertAlmostEqual(0.09, header.livetime.errorOfSeconds, places=3)
        assert 0 == header.coord_origin.x
        assert 0 == header.coord_origin.y
        assert 0 == header.coord_origin.z
        self.assertTupleEqual((0, 0, 0), header.coord_origin)


class TestConvertHeaderDictToTable(TestCase):
    def setUp(self):
        hdict = {
            "param_a": {
                "field_a_1": "1",
                "field_a_2": "2"
            },
            "param_b": {
                "field_b_1": "a"
            },
            "param_c": {
                "field_c_1": 1
            }
        }
        self.tab = convert_header_dict_to_table(hdict)

    def test_length(self):
        assert 3 == len(self.tab)

    def test_values(self):
        tab = self.tab

        index_a = tab.parameter.tolist().index("param_a")
        index_b = tab.parameter.tolist().index("param_b")

        assert "param_a" == tab.parameter[index_a]
        assert "field_a_1" in tab.field_names[index_a]
        assert "field_a_2" in tab.field_names[index_a]
        if "field_a_1 field_a_2" == tab.field_names[index_a]:
            assert "1 2" == tab.field_values[index_a]
        else:
            assert "2 1" == tab.field_values[index_a]
        assert "f4 f4" == tab['dtype'][index_a]

        assert "param_b" == tab.parameter[index_b]
        assert "field_b_1" == tab.field_names[index_b]
        assert "a" == tab.field_values[index_b]
        assert "a1" == tab['dtype'][index_b]

    def test_values_are_converted_to_str(self):
        index_c = self.tab.parameter.tolist().index("param_c")
        assert "param_c" == self.tab.parameter[index_c]
        assert "1" == self.tab.field_values[index_c]

    def test_conversion_returns_none_for_empty_dict(self):
        assert None is convert_header_dict_to_table(None)
        assert None is convert_header_dict_to_table({})
