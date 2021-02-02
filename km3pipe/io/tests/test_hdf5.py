#!/usr/bin/env python
"""Tests for HDF5 stuff"""
from collections import OrderedDict, defaultdict
import tempfile
from os.path import join, dirname

import numpy as np
import tables as tb
import km3io

from km3pipe import Blob, Module, Pipeline, version
from km3pipe.dataclasses import Table, NDArray
from km3pipe.io.hdf5 import (
    HDF5Pump,
    HDF5Sink,
    HDF5Header,
    header2table,
    FORMAT_VERSION,
)
from km3pipe.testing import TestCase, data_path


class Skipper(Module):
    """Skips the iteration with a given index (starting at 0)"""

    def configure(self):
        self.skip_indices = self.require("indices")
        self.index = 0

    def process(self, blob):
        self.index += 1
        if self.index - 1 in self.skip_indices:
            print("skipping")
            return
        print(blob)
        return blob


class TestH5Pump(TestCase):
    def setUp(self):
        self.fname = data_path(
            "hdf5/mcv5.40.mupage_10G.sirene.jterbr00006060.962.root.h5"
        )

    def test_init_sets_filename_if_no_keyword_arg_is_passed(self):
        p = HDF5Pump(filename=self.fname)
        self.assertEqual(self.fname, p.filename)
        p.finish()

    def test_standalone(self):
        pump = HDF5Pump(filename=self.fname)
        next(pump)
        pump.finish()

    def test_pipe(self):
        class Observer(Module):
            def configure(self):
                self.dump = defaultdict(list)

            def process(self, blob):
                for key, data in blob.items():
                    if key == "Header":
                        self.dump["headers"].append(data)
                    else:
                        self.dump[key].append(len(data))
                return blob

            def finish(self):
                return self.dump

        p = Pipeline()
        p.attach(HDF5Pump, filename=self.fname)
        p.attach(Observer)
        results = p.drain()["Observer"]
        self.assertListEqual(
            [147, 110, 70, 62, 59, 199, 130, 92, 296, 128], results["Hits"]
        )
        self.assertListEqual(
            [315, 164, 100, 111, 123, 527, 359, 117, 984, 263], results["McHits"]
        )
        self.assertListEqual([1, 1, 1, 1, 1, 3, 2, 1, 2, 1], results["McTracks"])

    def test_event_info_is_not_empty(self):
        self.fname = data_path("hdf5/test_event_info.h5")

        class Printer(Module):
            def process(self, blob):
                assert blob["EventInfo"].size != 0
                return blob

        p = Pipeline()
        p.attach(HDF5Pump, filename=self.fname)
        p.attach(Printer)
        p.drain()

    def test_event_info_has_correct_group_id(self):
        self.fname = data_path("hdf5/test_event_info.h5")

        class Printer(Module):
            def configure(self):
                self.index = 0

            def process(self, blob):
                assert blob["EventInfo"][0].group_id == self.index
                self.index += 1
                return blob

        p = Pipeline()
        p.attach(HDF5Pump, filename=self.fname)
        p.attach(Printer)
        p.drain()

    def test_get_blob(self):
        fname = data_path("hdf5/test_event_info.h5")
        pump = HDF5Pump(filename=fname)
        assert 44 == len(pump[0]["McTracks"])
        assert 3 == len(pump[1]["McTracks"])
        assert 179 == len(pump[2]["McTracks"])
        assert 55 == len(pump[3]["McTracks"])
        pump.finish()


class TestH5Sink(TestCase):
    def setUp(self):
        self.fname = data_path(
            "hdf5/mcv5.40.mupage_10G.sirene.jterbr00006060.962.root.h5"
        )
        self.fobj = tempfile.NamedTemporaryFile(delete=True)
        self.out = tb.open_file(
            self.fobj.name, "w", driver="H5FD_CORE", driver_core_backing_store=0
        )

    def tearDown(self):
        self.out.close()
        self.fobj.close()

    def test_pipe(self):
        p = Pipeline()
        p.attach(HDF5Pump, filename=self.fname)
        p.attach(HDF5Sink, h5file=self.out)
        p.drain()

    def test_h5info(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Module):
            def process(self, blob):
                return Blob()

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        with tb.open_file(fname, "r") as h5file:
            assert version == h5file.root._v_attrs.km3pipe.decode()
            assert tb.__version__ == h5file.root._v_attrs.pytables.decode()
            assert FORMAT_VERSION == h5file.root._v_attrs.format_version

        fobj.close()

    def test_filtered_writing(self):
        fobjs = []
        for i in range(3):
            fobj = tempfile.NamedTemporaryFile(delete=True)
            fobjs.append(fobj)

        fobj_all = tempfile.NamedTemporaryFile(delete=True)

        class DummyPump(Module):
            def configure(self):
                self.i = 0

            def process(self, blob):
                blob["A"] = Table({"a": self.i}, name="A", h5loc="tab_a")
                blob["B"] = Table({"b": self.i}, name="B", h5loc="tab_b")
                blob["C"] = Table({"c": self.i}, name="C", h5loc="tab_c")
                self.i += 1
                return blob

        keys = "ABC"

        pipe = Pipeline()
        pipe.attach(DummyPump)
        for fobj, key in zip(fobjs, keys):
            pipe.attach(HDF5Sink, filename=fobj.name, keys=[key])
        pipe.attach(HDF5Sink, filename=fobj_all.name)
        pipe.drain(5)

        for fobj, key in zip(fobjs, keys):
            with tb.File(fobj.name, "r") as f:
                assert "/tab_" + key.lower() in f
                for _key in set(keys) - set(key):
                    assert "/tab_" + _key.lower() not in f

        for key in keys:
            with tb.File(fobj_all.name, "r") as f:
                assert "/tab_" + key.lower() in f

        for fobj in fobjs:
            fobj.close()
        fobj_all.close()

    def test_filtered_writing_of_multiple_keys(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)

        class DummyPump(Module):
            def configure(self):
                self.i = 0

            def process(self, blob):
                blob["A"] = Table({"a": self.i}, name="A", h5loc="tab_a")
                blob["B"] = Table({"b": self.i}, name="B", h5loc="tab_b")
                blob["C"] = Table({"c": self.i}, name="C", h5loc="tab_c")
                self.i += 1
                return blob

        keys = ["A", "B"]

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fobj.name, keys=keys)
        pipe.drain(5)

        with tb.File(fobj.name, "r") as f:
            assert "/tab_a" in f
            assert "/tab_b" in f
            assert "/tab_c" not in f

        fobj.close()

    def test_write_table_service(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)

        class Foo(Module):
            def prepare(self):
                self.services["write_table"](Table({"a": 1}, name="A", h5loc="tab_a"))

        pipe = Pipeline()
        pipe.attach(Foo)
        pipe.attach(HDF5Sink, filename=fobj.name)
        pipe.drain(5)

        with tb.File(fobj.name, "r") as f:
            assert "/tab_a" in f

        fobj.close()


class TestNDArrayHandling(TestCase):
    def test_writing_of_n_dim_arrays_with_defaults(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        class DummyPump(Module):
            def process(self, blob):
                blob["foo"] = NDArray(arr)
                return blob

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(3)

        with tb.File(fname) as f:
            foo = f.get_node("/misc")
            assert 3 == foo[0, 1, 0]
            assert 4 == foo[0, 1, 1]
            assert "Unnamed NDArray" == foo.title
            indices = f.get_node("/misc_indices")
            self.assertTupleEqual((0, 2, 4), tuple(indices.cols.index[:]))
            self.assertTupleEqual((2, 2, 2), tuple(indices.cols.n_items[:]))

        fobj.close()

    def test_writing_of_n_dim_arrays(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        class DummyPump(Module):
            def configure(self):
                self.index = 0

            def process(self, blob):
                blob["foo"] = NDArray(arr + self.index * 10, h5loc="/foo", title="Yep")
                self.index += 1
                return blob

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(3)

        with tb.File(fname) as f:
            foo = f.get_node("/foo")
            assert 3 == foo[0, 1, 0]
            assert 4 == foo[0, 1, 1]
            assert "Yep" == foo.title

        fobj.close()

    def test_writing_of_n_dim_arrays_in_nested_group(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        class DummyPump(Module):
            def configure(self):
                self.index = 0

            def process(self, blob):
                blob["foo"] = NDArray(arr + self.index * 10, h5loc="/foo/bar/baz")
                self.index += 1
                return blob

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(3)

        with tb.File(fname) as f:
            foo = f.get_node("/foo/bar/baz")
            print(foo)
            assert 3 == foo[0, 1, 0]
            assert 4 == foo[0, 1, 1]

        fobj.close()

    def test_writing_of_n_dim_arrays(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        class DummyPump(Module):
            def configure(self):
                self.index = 0

            def process(self, blob):
                blob["foo"] = NDArray(arr + self.index * 10, h5loc="/foo", title="Yep")
                self.index += 1
                return blob

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(3)

        with tb.File(fname) as f:
            foo = f.get_node("/foo")
            assert 3 == foo[0, 1, 0]
            assert 4 == foo[0, 1, 1]
            assert "Yep" == foo.title

        fobj.close()

    def test_reading_of_n_dim_arrays(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        class DummyPump(Module):
            def configure(self):
                self.index = 0

            def process(self, blob):
                blob["Foo"] = NDArray(arr + self.index * 10, h5loc="/foo", title="Yep")
                self.index += 1
                return blob

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(3)

        class Observer(Module):
            def configure(self):
                self.index = 0

            def process(self, blob):
                assert "Foo" in blob
                foo = blob["Foo"]
                print(self.index)
                assert self.index * 10 + 1 == foo[0, 0, 0]
                assert self.index * 10 + 8 == foo[1, 1, 1]
                assert self.index * 10 + 3 == foo[0, 1, 0]
                assert self.index * 10 + 6 == foo[1, 0, 1]
                self.index += 1
                return blob

        pipe = Pipeline()
        pipe.attach(HDF5Pump, filename=fname)
        pipe.attach(Observer)
        pipe.drain()

        fobj.close()


class TestH5SinkSkippedBlobs(TestCase):
    def test_skipped_blob_with_tables(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Module):
            def configure(self):
                self.index = 0

            def process(self, blob):
                blob["Tab"] = Table(
                    {"a": np.arange(self.index + 1), "i": self.index}, h5loc="/tab"
                )
                self.index += 1
                return blob

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(Skipper, indices=[2])
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        with tb.File(fname) as f:
            a = f.get_node("/tab")[:]["a"]
            i = f.get_node("/tab")[:]["i"]
            group_id = f.get_node("/tab")[:]["group_id"]
        assert np.allclose([0, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4, 4], i)
        assert np.allclose([0, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4], a)
        assert np.allclose([0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3], group_id)

        fobj.close()

    def test_skipped_blob_with_ndarray(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Module):
            def configure(self):
                self.index = 0

            def process(self, blob):
                blob["Arr"] = NDArray(np.arange(self.index + 1), h5loc="/arr")
                self.index += 1
                return blob

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(Skipper, indices=[2])
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        with tb.File(fname) as f:
            a = f.get_node("/arr")[:]
            index_table = f.get_node("/arr_indices")[:]
        assert np.allclose([0, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4], a)
        assert np.allclose([0, 1, 3, 7], index_table["index"])
        assert np.allclose([1, 2, 4, 5], index_table["n_items"])

        fobj.close()

    def test_skipped_blob_with_tables_and_ndarrays(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Module):
            def configure(self):
                self.index = 0

            def process(self, blob):
                blob["Arr"] = NDArray(np.arange(self.index + 1), h5loc="/arr")
                blob["Tab"] = Table(
                    {"a": np.arange(self.index + 1), "i": self.index}, h5loc="/tab"
                )
                self.index += 1
                return blob

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(Skipper, indices=[2])
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        with tb.File(fname) as f:
            tab_a = f.get_node("/tab")[:]["a"]
            tab_i = f.get_node("/tab")[:]["i"]
            group_id = f.get_node("/tab")[:]["group_id"]

            arr = f.get_node("/arr")[:]
            index_table = f.get_node("/arr_indices")[:]

            group_info = f.get_node("/group_info")[:]

        assert np.allclose([0, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4, 4], tab_i)
        assert np.allclose([0, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4], tab_a)
        assert np.allclose([0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3], group_id)

        assert np.allclose([0, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4], arr)
        assert np.allclose([0, 1, 3, 7], index_table["index"])
        assert np.allclose([1, 2, 4, 5], index_table["n_items"])

        fobj.close()

    def test_skipped_blob_with_tables_and_ndarrays_first_and_last(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Module):
            def configure(self):
                self.index = 0

            def process(self, blob):
                blob["Arr"] = NDArray(np.arange(self.index + 1), h5loc="/arr")
                blob["Tab"] = Table(
                    {"a": np.arange(self.index + 1), "i": self.index}, h5loc="/tab"
                )
                self.index += 1
                return blob

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(Skipper, indices=[0, 4])
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        with tb.File(fname) as f:
            tab_a = f.get_node("/tab")[:]["a"]
            tab_i = f.get_node("/tab")[:]["i"]
            group_id = f.get_node("/tab")[:]["group_id"]

            arr = f.get_node("/arr")[:]
            index_table = f.get_node("/arr_indices")[:]

            group_info = f.get_node("/group_info")[:]

        assert np.allclose([1, 1, 2, 2, 2, 3, 3, 3, 3], tab_i)
        assert np.allclose([0, 1, 0, 1, 2, 0, 1, 2, 3], tab_a)
        assert np.allclose([0, 0, 1, 1, 1, 2, 2, 2, 2], group_id)

        assert np.allclose([0, 1, 0, 1, 2, 0, 1, 2, 3], arr)
        assert np.allclose([0, 2, 5], index_table["index"])
        assert np.allclose([2, 3, 4], index_table["n_items"])

        fobj.close()


class TestH5SinkConsistency(TestCase):
    def test_h5_consistency_for_tables_without_group_id(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Module):
            def configure(self):
                self.count = 0

            def process(self, blob):
                self.count += 10
                tab = Table({"a": self.count, "b": 1}, h5loc="tab")
                return Blob({"tab": tab})

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        with tb.File(fname) as f:
            a = f.get_node("/tab")[:]["a"]
            b = f.get_node("/tab")[:]["b"]
            group_id = f.get_node("/tab")[:]["group_id"]
        assert np.allclose([10, 20, 30, 40, 50], a)
        assert np.allclose([1, 1, 1, 1, 1], b)
        assert np.allclose([0, 1, 2, 3, 4], group_id)
        fobj.close()

    def test_h5_consistency_for_tables_without_group_id_and_multiple_keys(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Module):
            def configure(self):
                self.count = 0

            def process(self, blob):
                self.count += 10
                tab1 = Table({"a": self.count, "b": 1}, h5loc="tab1")
                tab2 = Table({"c": self.count + 1, "d": 2}, h5loc="tab2")
                return Blob({"tab1": tab1, "tab2": tab2})

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        with tb.File(fname) as f:
            a = f.get_node("/tab1")[:]["a"]
            b = f.get_node("/tab1")[:]["b"]
            c = f.get_node("/tab2")[:]["c"]
            d = f.get_node("/tab2")[:]["d"]
            group_id_1 = f.get_node("/tab1")[:]["group_id"]
            group_id_2 = f.get_node("/tab1")[:]["group_id"]
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

        class DummyPump(Module):
            def process(self, blob):
                tab = Table({"group_id": 2}, h5loc="tab")
                return Blob({"tab": tab})

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname, reset_group_id=False)
        pipe.drain(5)

        with tb.File(fname) as f:
            group_id = f.get_node("/tab")[:]["group_id"]

        assert np.allclose([2, 2, 2, 2, 2], group_id)

        fobj.close()

    def test_h5_singletons(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Module):
            def process(self, blob):
                tab = Table({"a": 2}, h5loc="tab", h5singleton=True)
                return Blob({"tab": tab})

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        with tb.File(fname) as f:
            a = f.get_node("/tab")[:]["a"]

        assert len(a) == 1

        fobj.close()

    def test_h5_singletons_reading(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Module):
            def process(self, blob):
                tab = Table({"a": 2}, h5loc="tab", h5singleton=True)
                return Blob({"Tab": tab})

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        class Observer(Module):
            def process(self, blob):
                print(blob)
                assert "Tab" in blob
                print(blob["Tab"])
                assert len(blob["Tab"]) == 1
                assert blob["Tab"].a[0] == 2
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

        class DummyPump(Module):
            def configure(self):
                self.count = 0

            def process(self, blob):
                self.count += 1
                tab = Table({"a": self.count * 10, "b": 1}, h5loc="tab")
                tab2 = Table({"a": np.arange(self.count)}, h5loc="tab2")
                blob["Tab"] = tab
                blob["Tab2"] = tab2
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
                assert "GroupInfo" in blob
                assert "Tab" in blob
                print(self.index)
                print(blob["Tab"])
                print(blob["Tab"]["a"])
                assert self.index - 1 == blob["GroupInfo"].group_id
                assert self.index * 10 == blob["Tab"]["a"]
                assert 1 == blob["Tab"]["b"] == 1
                assert np.allclose(np.arange(self.index), blob["Tab2"]["a"])
                return blob

        pipe = Pipeline()
        pipe.attach(HDF5Pump, filename=fname)
        pipe.attach(BlobTester)
        pipe.drain()

        fobj.close()

    def test_hdf5_readout_split_tables(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Module):
            def configure(self):
                self.count = 0

            def process(self, blob):
                self.count += 1
                tab = Table({"a": self.count * 10, "b": 1}, h5loc="/tab", split_h5=True)
                blob["Tab"] = tab
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
                assert "GroupInfo" in blob
                assert "Tab" in blob
                assert self.index - 1 == blob["GroupInfo"].group_id
                assert self.index * 10 == blob["Tab"]["a"]
                assert 1 == blob["Tab"]["b"]
                return blob

        pipe = Pipeline()
        pipe.attach(HDF5Pump, filename=fname)
        pipe.attach(BlobTester)
        pipe.drain()

        fobj.close()

    def test_hdf5_readout_split_tables_in_same_group(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Module):
            def configure(self):
                self.count = 0

            def process(self, blob):
                self.count += 1
                tab_a = Table(
                    {
                        "a": self.count * 10,
                    },
                    h5loc="/tabs/tab_a",
                    split_h5=True,
                )
                tab_b = Table(
                    {
                        "b": self.count * 100,
                    },
                    h5loc="/tabs/tab_b",
                    split_h5=True,
                )
                blob["TabA"] = tab_a
                blob["TabB"] = tab_b
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
                assert "GroupInfo" in blob
                assert "TabA" in blob
                assert "TabB" in blob
                assert self.index - 1 == blob["GroupInfo"].group_id
                assert self.index * 10 == blob["TabA"]["a"]
                assert self.index * 100 == blob["TabB"]["b"]
                return blob

        pipe = Pipeline()
        pipe.attach(HDF5Pump, filename=fname)
        pipe.attach(BlobTester)
        pipe.drain()

        fobj.close()

    def test_sparse_table(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class Dummy(Module):
            def configure(self):
                self.i = 0

            def process(self, blob):
                self.i += 1

                if self.i == 5:
                    blob["Tab"] = Table({"a": 23}, h5loc="/tab")
                return blob

        pipe = Pipeline()
        pipe.attach(Dummy)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(10)

        class Observer(Module):
            def configure(self):
                self.i = 0

            def process(self, blob):
                self.i += 1

                if self.i == 5:
                    assert 23 == blob["Tab"].a[0]
                else:
                    assert "Tab" not in blob

                return blob

        pipe = Pipeline()
        pipe.attach(HDF5Pump, filename=fname)
        pipe.attach(Observer)
        pipe.drain()

    def test_sparse_ndarray(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class Dummy(Module):
            def configure(self):
                self.i = 0

            def process(self, blob):
                self.i += 1

                if self.i == 5:
                    blob["Arr"] = NDArray([1, 2, 3], h5loc="/arr")
                return blob

        pipe = Pipeline()
        pipe.attach(Dummy)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(10)

        class Observer(Module):
            def configure(self):
                self.i = 0

            def process(self, blob):
                self.i += 1

                print(blob)
                if self.i == 5:
                    assert 6 == np.sum(blob["Arr"])
                else:
                    assert len(blob["Arr"]) == 0

                return blob

        pipe = Pipeline()
        pipe.attach(HDF5Pump, filename=fname)
        pipe.attach(Observer)
        pipe.drain()


class TestHDF5Shuffle(TestCase):
    def test_shuffle_without_reset_index(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Module):
            def configure(self):
                self.i = 0

            def process(self, blob):
                blob["Tab"] = Table({"a": self.i}, h5loc="/tab")
                blob["SplitTab"] = Table(
                    {"b": self.i}, h5loc="/split_tab", split_h5=True
                )
                blob["Arr"] = NDArray(np.arange(self.i + 1), h5loc="/arr")
                self.i += 1
                return blob

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        shuffled_group_ids = [2, 1, 0, 3, 4]

        def shuffle(x):
            for i in range(len(x)):
                x[i] = shuffled_group_ids[i]

        class Observer(Module):
            def configure(self):
                self.group_ids_tab = []
                self.group_ids_split_tab = []
                self.group_ids_arr = []
                self.a = []
                self.b = []
                self.arr_len = []

            def process(self, blob):
                group_id_tab = blob["Tab"].group_id[0]
                group_id_split_tab = blob["SplitTab"].group_id[0]
                group_id_arr = blob["Arr"].group_id
                assert blob["GroupInfo"].group_id[0] == group_id_tab
                assert blob["GroupInfo"].group_id[0] == group_id_split_tab
                assert blob["GroupInfo"].group_id[0] == group_id_arr
                self.group_ids_tab.append(blob["Tab"].group_id[0])
                self.group_ids_split_tab.append(blob["SplitTab"].group_id[0])
                self.group_ids_arr.append(blob["Arr"].group_id)
                self.a.append(blob["Tab"].a[0])
                self.b.append(blob["SplitTab"].b[0])
                self.arr_len.append(len(blob["Arr"]) - 1)
                return blob

            def finish(self):
                return {
                    "group_ids_tab": self.group_ids_tab,
                    "group_ids_split_tab": self.group_ids_split_tab,
                    "group_ids_arr": self.group_ids_arr,
                    "a": self.a,
                    "b": self.b,
                    "arr_len": self.arr_len,
                }

        pipe = Pipeline()
        pipe.attach(
            HDF5Pump,
            filename=fname,
            shuffle=True,
            shuffle_function=shuffle,
            reset_index=False,
        )
        pipe.attach(Observer)
        results = pipe.drain()

        self.assertListEqual(results["Observer"]["group_ids_tab"], shuffled_group_ids)
        self.assertListEqual(
            results["Observer"]["group_ids_split_tab"], shuffled_group_ids
        )
        self.assertListEqual(results["Observer"]["group_ids_arr"], shuffled_group_ids)
        self.assertListEqual(results["Observer"]["a"], shuffled_group_ids)
        self.assertListEqual(results["Observer"]["b"], shuffled_group_ids)
        # a small hack: we store the length of the array in 'b', which is
        # then equal to the shuffled group IDs (since those were generated
        # using the group_id
        self.assertListEqual(results["Observer"]["arr_len"], shuffled_group_ids)

        fobj.close()

    def test_shuffle_with_reset_index(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = fobj.name

        class DummyPump(Module):
            def configure(self):
                self.i = 0

            def process(self, blob):
                blob["Tab"] = Table({"a": self.i}, h5loc="/tab")
                blob["SplitTab"] = Table(
                    {"b": self.i}, h5loc="/split_tab", split_h5=True
                )
                blob["Arr"] = NDArray(np.arange(self.i + 1), h5loc="/arr")
                self.i += 1
                return blob

        pipe = Pipeline()
        pipe.attach(DummyPump)
        pipe.attach(HDF5Sink, filename=fname)
        pipe.drain(5)

        shuffled_group_ids = [2, 1, 0, 3, 4]

        def shuffle(x):
            for i in range(len(x)):
                x[i] = shuffled_group_ids[i]

        class Observer(Module):
            def configure(self):
                self.group_ids_tab = []
                self.group_ids_split_tab = []
                self.group_ids_arr = []
                self.a = []
                self.b = []
                self.arr_len = []

            def process(self, blob):
                group_id_tab = blob["Tab"].group_id[0]
                group_id_split_tab = blob["SplitTab"].group_id[0]
                group_id_arr = blob["Arr"].group_id
                assert blob["GroupInfo"].group_id[0] == group_id_tab
                assert blob["GroupInfo"].group_id[0] == group_id_split_tab
                assert blob["GroupInfo"].group_id[0] == group_id_arr
                self.group_ids_tab.append(blob["Tab"].group_id[0])
                self.group_ids_split_tab.append(blob["SplitTab"].group_id[0])
                self.group_ids_arr.append(blob["Arr"].group_id)
                self.a.append(blob["Tab"].a[0])
                self.b.append(blob["SplitTab"].b[0])
                self.arr_len.append(len(blob["Arr"]) - 1)
                return blob

            def finish(self):
                return {
                    "group_ids_tab": self.group_ids_tab,
                    "group_ids_split_tab": self.group_ids_split_tab,
                    "group_ids_arr": self.group_ids_arr,
                    "a": self.a,
                    "b": self.b,
                    "arr_len": self.arr_len,
                }

        pipe = Pipeline()
        pipe.attach(
            HDF5Pump,
            filename=fname,
            shuffle=True,
            shuffle_function=shuffle,
            reset_index=True,
        )
        pipe.attach(Observer)
        results = pipe.drain()

        self.assertListEqual(results["Observer"]["group_ids_tab"], [0, 1, 2, 3, 4])
        self.assertListEqual(
            results["Observer"]["group_ids_split_tab"], [0, 1, 2, 3, 4]
        )
        self.assertListEqual(results["Observer"]["group_ids_arr"], [0, 1, 2, 3, 4])
        self.assertListEqual(results["Observer"]["a"], shuffled_group_ids)
        self.assertListEqual(results["Observer"]["b"], shuffled_group_ids)
        # a small hack: we store the length of the array in 'b', which is
        # then equal to the shuffled group IDs (since those were generated
        # using the group_id
        self.assertListEqual(results["Observer"]["arr_len"], shuffled_group_ids)

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
            "param_a": {"field_a_1": "1", "field_a_2": "2"},
            "param_b": {"field_b_1": "a"},
            "param_c": {"field_c_1": 23},
            "param_d": {"param_d_0": 1, "param_d_1": 2, "param_d_2": 3},
            "param_e": {"param_e_2": 3, "param_e_0": 1, "param_e_1": 2},
            # "param+invalid.attribute": {"a": 1, "b": 2, "c": 3}
        }

    def test_init(self):
        HDF5Header({})

    def test_header_behaves_like_a_dict(self):
        h = HDF5Header(self.hdict)
        self.assertListEqual(list(h.keys()), list(self.hdict.keys()))
        assert 5 == len(h.items())
        assert 5 == len(h.values())

    def test_header(self):
        header = HDF5Header(self.hdict)
        assert "1" == header.param_a.field_a_1
        assert "2" == header.param_a.field_a_2
        assert "a" == header.param_b.field_b_1
        assert 23 == header.param_c.field_c_1

    def test_header_getitem(self):
        header = HDF5Header(self.hdict)
        print(header["param_a"])
        assert "1" == header["param_a"].field_a_1
        assert "2" == header["param_a"].field_a_2
        assert "a" == header["param_b"].field_b_1
        assert 23 == header["param_c"].field_c_1

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
        table = header2table(self.hdict)
        header = HDF5Header.from_table(table)
        print(header)
        assert 1.0 == header.param_a.field_a_1
        assert 2.0 == header.param_a.field_a_2
        assert "a" == header.param_b.field_b_1
        assert 23 == header.param_c.field_c_1
        self.assertTupleEqual((1, 2, 3), header.param_d)

    def test_header_from_hdf5_file(self):
        header = HDF5Header.from_hdf5(data_path("hdf5/raw_header.h5"))
        assert "MUSIC" == header.propag[0]
        assert "seawater" == header.propag[1]
        assert 3450 == header.seabottom[0]
        self.assertAlmostEqual(12.1, header.livetime.numberOfSeconds, places=3)
        self.assertAlmostEqual(0.09, header.livetime.errorOfSeconds, places=3)
        assert 0 == header.coord_origin.x
        assert 0 == header.coord_origin.y
        assert 0 == header.coord_origin.z
        self.assertTupleEqual((0, 0, 0), header.coord_origin)

    def test_header_from_hdf5_file_with_invalid_identifier_names_in_header(self):
        header = HDF5Header.from_hdf5(data_path("hdf5/geamon.h5"))
        assert 1.0 == header["drays+z"][0]
        assert 68.5 == header["drays+z"][1]

    def test_header_from_table_with_bytes(self):
        table = Table(
            {
                "dtype": [b"f4 a2", b"f4"],
                "field_names": [b"a b", b"c"],
                "field_values": [b"1.2 ab", b"3.4"],
                "parameter": [b"foo", b"bar"],
            }
        )
        header = HDF5Header.from_aanet(table)
        self.assertAlmostEqual(1.2, header.foo.a, places=2)
        assert "ab" == header.foo.b
        self.assertAlmostEqual(3.4, header.bar.c, places=2)

    def test_header_from_km3io(self):
        head = {
            "a": "1 2 3",
            "b+c": "4 5 6",
            "c": "foo",
            "d": "7",
            "e+f": "bar",
        }

        header = HDF5Header.from_km3io(km3io.offline.Header(head))

        assert 1 == header["a"][0]
        assert 2 == header["a"][1]
        assert 3 == header["a"][2]
        assert 1 == header.a[0]
        assert 2 == header.a[1]
        assert 3 == header.a[2]
        assert 4 == header["b+c"][0]
        assert 5 == header["b+c"][1]
        assert 6 == header["b+c"][2]
        assert "foo" == header.c
        assert "foo" == header["c"]
        assert 7 == header.d
        assert 7 == header["d"]
        assert "bar" == header["e+f"]


class TestConvertHeaderDictToTable(TestCase):
    def setUp(self):
        hdict = {
            "param_a": {"field_a_1": "1", "field_a_2": "2"},
            "param_b": {"field_b_1": "a"},
            "param_c": {"field_c_1": 1},
        }
        self.tab = header2table(hdict)

    def test_length(self):
        assert 3 == len(self.tab)

    def test_values(self):
        tab = self.tab

        index_a = tab.parameter.tolist().index(b"param_a")
        index_b = tab.parameter.tolist().index(b"param_b")

        assert b"param_a" == tab.parameter[index_a]
        assert b"field_a_1" in tab.field_names[index_a]
        assert b"field_a_2" in tab.field_names[index_a]
        if b"field_a_1 field_a_2" == tab.field_names[index_a]:
            assert b"1 2" == tab.field_values[index_a]
        else:
            assert b"2 1" == tab.field_values[index_a]
        assert b"f4 f4" == tab["dtype"][index_a]

        assert b"param_b" == tab.parameter[index_b]
        assert b"field_b_1" == tab.field_names[index_b]
        assert b"a" == tab.field_values[index_b]
        assert b"a1" == tab["dtype"][index_b]

    def test_values_are_converted_to_str(self):
        index_c = self.tab.parameter.tolist().index(b"param_c")
        assert b"param_c" == self.tab.parameter[index_c]
        assert b"1" == self.tab.field_values[index_c]

    def test_conversion_returns_none_for_empty_dict(self):
        assert None is header2table(None)
        assert None is header2table({})

    def test_conversion_of_km3io_header(self):
        header = km3io.OfflineReader(data_path("offline/numucc.root")).header
        tab = header2table(header)
        print(tab)
        for p in [
            b"DAQ",
            b"PDF",
            b"can",
            b"can_user",
            b"coord_origin",
            b"cut_in",
            b"cut_nu",
            b"cut_primary",
            b"cut_seamuon",
            b"decay",
            b"detector",
            b"drawing",
            b"genhencut",
            b"genvol",
            b"kcut",
            b"livetime",
            b"model",
            b"ngen",
            b"norma",
            b"nuflux",
            b"physics",
            b"seed",
            b"simul",
            b"sourcemode",
            b"spectrum",
            b"start_run",
            b"target",
            b"usedetfile",
            b"xlat_user",
            b"xparam",
            b"zed_user",
        ]:
            assert p in tab.parameter

        h5header = HDF5Header.from_table(tab)
        assert h5header.can.zmin == header.can.zmin

    def test_conversion_of_hdf5header(self):
        hdict = {
            "param_a": {"field_a_1": "1", "field_a_2": "2"},
            "param_b": {"field_b_1": "a"},
            "param_c": {"field_c_1": 1},
        }

        header = HDF5Header(hdict)
        tab = header2table(header)

        for p in [b"param_a", b"param_b", b"param_c"]:
            assert p in tab.parameter

        hdf5header_from_table = HDF5Header.from_table(tab)
