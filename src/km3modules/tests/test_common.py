# Filename: test_time.py
# pylint: disable=locally-disabled,C0111,R0904,C0103

import sqlite3
import tempfile

import km3pipe as kp
from km3pipe.dataclasses import Table
from km3modules.common import (
    Siphon,
    Delete,
    Keep,
    Dump,
    StatusBar,
    TickTock,
    MemoryObserver,
    BlobIndexer,
    LocalDBService,
    Observer,
    MultiFilePump,
    FilePump,
)
from km3pipe.testing import TestCase, MagicMock
from km3pipe.tools import istype

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class InfinitePump(kp.Module):
    """A pump which just infinetly spits out indexed blobs"""

    def configure(self):
        self.i = 0

    def process(self, blob):
        self.i += 1
        blob["i"] = self.i
        return blob


class TestKeep(TestCase):
    def test_keep_a_single_key(self):
        class APump(kp.Module):
            def process(self, blob):
                blob["a"] = "a"
                blob["b"] = "b"
                blob["c"] = "c"
                blob["d"] = "d"
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert "a" not in blob
                assert "b" not in blob
                assert "c" not in blob
                assert "d" == blob["d"]
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Keep, keys="d")
        pipe.attach(Observer)
        pipe.drain(5)

    def test_keep_multiple_keys(self):
        class APump(kp.Module):
            def process(self, blob):
                blob["a"] = "a"
                blob["b"] = "b"
                blob["c"] = "c"
                blob["d"] = "d"
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert "a" not in blob
                assert "b" == blob["b"]
                assert "c" not in blob
                assert "d" == blob["d"]
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Keep, keys=["b", "d"])
        pipe.attach(Observer)
        pipe.drain(5)

    def test_hdf5_keep_group_wo_subgroup(self):
        class APump(kp.Module):
            def process(self, blob):
                blob["A"] = kp.Table(
                    {"foo": [1, 2, 3], "bar": [4, 5, 6]}, h5loc="/foobar"
                )
                blob["B"] = kp.Table(
                    {"a": [1.1, 2.1, 3.1], "b": [4.2, 5.2, 6.2]}, h5loc="/ab"
                )
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert "A" in blob.keys()
                assert "/foobar" == blob["A"].h5loc
                assert not "B" in blob.keys()
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Keep, h5locs=["/foobar"])
        pipe.attach(Observer)
        pipe.drain(5)

    def test_hdf5_keep_group_w_subgroup(self):
        class APump(kp.Module):
            def process(self, blob):
                blob["A"] = kp.Table(
                    {"foo": [1, 2, 3], "bar": [4, 5, 6]}, h5loc="/foobar"
                )
                blob["B"] = kp.Table(
                    {"a": [1.1, 2.1, 3.1], "b": [4.2, 5.2, 6.2]}, h5loc="/ab"
                )
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert "A" in blob.keys()
                assert "/foobar" == blob["A"].h5loc
                assert not "B" in blob.keys()
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Keep, h5locs=["/foobar"])
        pipe.attach(Observer)
        pipe.drain(5)

    def test_key_hdf5_group_individual(self):
        class APump(kp.Module):
            def process(self, blob):
                blob["A"] = kp.Table(
                    {"foo": [1, 2, 3], "bar": [4, 5, 6]}, h5loc="/foobar"
                )
                blob["B"] = kp.Table(
                    {"a": [1.1, 2.1, 3.1], "b": [4.2, 5.2, 6.2]}, h5loc="/ab"
                )
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert "A" in blob.keys()
                assert "/foobar" == blob["A"].h5loc
                assert "B" in blob.keys()
                assert "/ab" == blob["B"].h5loc
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Keep, keys=["B"], h5locs=["/foobar"])
        pipe.attach(Observer)
        pipe.drain(5)

    def test_key_hdf5_group_parallel(self):
        class APump(kp.Module):
            def process(self, blob):
                blob["A"] = kp.Table(
                    {"foo": [1, 2, 3], "bar": [4, 5, 6]}, h5loc="/foobar"
                )
                blob["B"] = kp.Table(
                    {"a": [1.1, 2.1, 3.1], "b": [4.2, 5.2, 6.2]}, h5loc="/ab"
                )
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert "A" in blob.keys()
                assert "/foobar" == blob["A"].h5loc
                assert not "B" in blob.keys()
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Keep, keys=["A"], h5locs=["/foobar"])
        pipe.attach(Observer)
        pipe.drain(5)

    def test_major_hdf5_group(self):
        class APump(kp.Module):
            def process(self, blob):
                blob["A"] = kp.Table(
                    {"foo": [1, 2, 3], "bar": [4, 5, 6]}, h5loc="/foobar/a"
                )
                blob["B"] = kp.Table(
                    {"a": [1.1, 2.1, 3.1], "b": [4.2, 5.2, 6.2]}, h5loc="/foobar/b"
                )
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert "A" in blob.keys()
                assert "/foobar/a" == blob["A"].h5loc
                assert "B" in blob.keys()
                assert "/foobar/b" == blob["B"].h5loc
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Keep, h5locs=["/foobar"])
        pipe.attach(Observer)
        pipe.drain(5)

    def test_major_hdf5_group_nested(self):
        class APump(kp.Module):
            def process(self, blob):
                blob["A"] = kp.Table({"a": 0}, h5loc="/foo/bar/a")
                blob["B"] = kp.Table({"b": 1}, h5loc="/foo/bar/baz/b")
                blob["C"] = kp.Table({"c": 2}, h5loc="/foo/bar/baz/fjord/c")
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert "A" not in blob
                assert "B" in blob
                assert "C" in blob
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Keep, h5locs=["/foo/bar/baz"])
        pipe.attach(Observer)
        pipe.drain(5)


class TestDelete(TestCase):
    def test_delete_a_single_key(self):
        class APump(kp.Module):
            def process(self, blob):
                blob["a"] = "a"
                blob["b"] = "b"
                blob["c"] = "c"
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert "a" == blob["a"]
                assert "b" not in blob
                assert "c" == blob["c"]
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Delete, key="b")
        pipe.attach(Observer)
        pipe.drain(5)

    def test_delete_multiple_keys(self):
        class APump(kp.Module):
            def process(self, blob):
                blob["a"] = "a"
                blob["b"] = "b"
                blob["c"] = "c"
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert "a" not in blob
                assert "b" not in blob
                assert "c" == blob["c"]
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Delete, keys=["a", "b"])
        pipe.attach(Observer)
        pipe.drain(5)


class TestSiphon(TestCase):
    def test_siphon(self):
        class Observer(kp.Module):
            def configure(self):
                self.mock = MagicMock()

            def process(self, blob):
                self.mock()
                return blob

            def finish(self):
                assert self.mock.call_count == 7

        pipe = kp.Pipeline()
        pipe.attach(InfinitePump)
        pipe.attach(Siphon, volume=10)
        pipe.attach(Observer)
        pipe.drain(17)

    def test_siphon_with_flush(self):
        class Observer(kp.Module):
            def configure(self):
                self.mock = MagicMock()

            def process(self, blob):
                self.mock()
                return blob

            def finish(self):
                assert self.mock.call_count == 1

        pipe = kp.Pipeline()
        pipe.attach(InfinitePump)
        pipe.attach(Siphon, volume=10, flush=True)
        pipe.attach(Observer)
        pipe.drain(21)

    def test_siphon_with_flush_2(self):
        class Observer(kp.Module):
            def configure(self):
                self.mock = MagicMock()

            def process(self, blob):
                self.mock()
                return blob

            def finish(self):
                assert self.mock.call_count == 2

        pipe = kp.Pipeline()
        pipe.attach(InfinitePump)
        pipe.attach(Siphon, volume=10, flush=True)
        pipe.attach(Observer)
        pipe.drain(22)


class TestDump(TestCase):
    def test_dump(self):
        def add_something(blob):
            blob["a"] = 1
            return blob

        pipe = kp.Pipeline()
        pipe.attach(InfinitePump)
        pipe.attach(add_something)
        pipe.attach(Dump)
        pipe.drain(3)

    def test_dump_a_key(self):
        def add_something(blob):
            blob["a"] = 1
            return blob

        pipe = kp.Pipeline()
        pipe.attach(InfinitePump)
        pipe.attach(add_something)
        pipe.attach(Dump, key="a")
        pipe.drain(3)

    def test_dump_multiple_keys(self):
        def add_something(blob):
            blob["a"] = 1
            blob["b"] = 2
            return blob

        pipe = kp.Pipeline()
        pipe.attach(InfinitePump)
        pipe.attach(add_something)
        pipe.attach(Dump, keys=["a", "b"])
        pipe.drain(3)

    def test_dump_full(self):
        def add_something(blob):
            blob["a"] = 1
            blob["b"] = 2
            return blob

        pipe = kp.Pipeline()
        pipe.attach(InfinitePump)
        pipe.attach(add_something)
        pipe.attach(Dump, full=True)
        pipe.drain(3)


class TestStatusbar(TestCase):
    def test_statusbar(self):
        pipe = kp.Pipeline()
        pipe.attach(InfinitePump)
        pipe.attach(StatusBar, every=2)
        pipe.drain(5)


class TestTickTock(TestCase):
    def test_ticktock(self):
        pipe = kp.Pipeline()
        pipe.attach(InfinitePump)
        pipe.attach(TickTock)
        pipe.drain(5)


class TestMemoryObserver(TestCase):
    def test_memory_observer(self):
        pipe = kp.Pipeline()
        pipe.attach(InfinitePump)
        pipe.attach(MemoryObserver)
        pipe.drain(5)


class TestBlobIndexer(TestCase):
    def test_blob_indexer(self):
        class Observer(kp.Module):
            def configure(self):
                self.index = 0

            def process(self, blob):
                assert blob["blob_index"] == self.index
                self.index += 1
                return blob

        pipe = kp.Pipeline()
        pipe.attach(InfinitePump)
        pipe.attach(BlobIndexer)
        pipe.attach(Observer)
        pipe.drain(4)


class TestLocalDBService(TestCase):
    def test_create_table(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        dbs = LocalDBService(filename=fobj.name)
        dbs.create_table("foo", ["a", "b"], ["INT", "TEXT"])
        assert dbs.table_exists("foo")

    def test_create_table_does_not_overwrite_by_default(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        dbs = LocalDBService(filename=fobj.name)
        dbs.create_table("foo", ["a", "b"], ["INT", "TEXT"])
        with self.assertRaises(sqlite3.OperationalError):
            dbs.create_table("foo", ["a", "b"], ["INT", "TEXT"])

    def test_create_table_allows_overwrite(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        dbs = LocalDBService(filename=fobj.name)
        dbs.create_table("foo", ["a", "b"], ["INT", "TEXT"])
        dbs.create_table("foo", ["a", "b"], ["INT", "TEXT"], overwrite=True)

    def test_insert_row(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        dbs = LocalDBService(filename=fobj.name)
        dbs.create_table("foo", ["a", "b"], ["INT", "TEXT"])

        dbs.insert_row("foo", ["a", "b"], (23, "42"))
        dbs.insert_row("foo", ["a", "b"], (5, "hello"))

        cur = dbs.connection.cursor()
        cur.execute("SELECT * FROM foo")
        data = cur.fetchall()
        assert 2 == len(data)
        assert 23 == data[0][0]
        assert "42" == data[0][1]
        assert 5 == data[1][0]
        assert "hello" == data[1][1]


class TestObserver(TestCase):
    def test_observer(self):
        class Dummy(kp.Module):
            def process(self, blob):
                blob["a"] = 1
                return blob

        pipe = kp.Pipeline()
        pipe.attach(Dummy)
        pipe.attach(Observer, count=5, required_keys="a")
        pipe.drain(5)

    def test_observer_raises_when_count_wrong(self):
        class Dummy(kp.Module):
            def process(self, blob):
                return blob

        pipe = kp.Pipeline()
        pipe.attach(Dummy)
        pipe.attach(Observer, count=5)

        with self.assertRaises(AssertionError):
            pipe.drain(2)

    def test_observer_raises_when_key_is_missing(self):
        class Dummy(kp.Module):
            def process(self, blob):
                blob["a"] = 1
                return blob

        pipe = kp.Pipeline()
        pipe.attach(Dummy)
        pipe.attach(Observer, required_keys=["b"])

        with self.assertRaises(AssertionError):
            pipe.drain(2)


class TestMultiFilePump(TestCase):
    def test_iteration(self):
        class DummyPump(kp.Module):
            def configure(self):
                self.idx = 0
                self.max_iterations = self.get("max_iterations", default=5)
                self.blobs = self.blob_generator()
                assert 23 == self.get("foo")
                assert "narf" == self.get("bar")

            def process(self, blob):
                return next(self)

            def blob_generator(self):
                for idx in range(self.max_iterations):
                    yield kp.Blob({"index": self.idx, "tab": kp.Table({"a": 1})})

            def finish(self):
                return self.idx

            def __iter__(self):
                return self

            def __next__(self):
                return next(self.blobs)

        filenames = ["a", "b", "c"]
        max_iterations = 5
        total_iterations = max_iterations * len(filenames)

        super_self = self

        class Observer(kp.Module):
            def configure(self):
                self.count = 0
                self.filenames = []
                self.group_id = []

            def process(self, blob):
                self.count += 1
                self.filenames.append(blob["filename"])
                self.group_id.append(blob["tab"].group_id[0])
                return blob

            def finish(self):
                assert self.count == total_iterations
                assert "".join(f * max_iterations for f in filenames) == "".join(
                    self.filenames
                )
                super_self.assertListEqual(list(range(total_iterations)), self.group_id)

        pipe = kp.Pipeline()
        pipe.attach(
            MultiFilePump,
            pump=DummyPump,
            filenames=filenames,
            max_iterations=max_iterations,
            kwargs={"foo": 23, "bar": "narf"},
        )
        pipe.attach(Observer)
        pipe.drain()


class TestFilePump(TestCase):
    def test_iteration(self):
        filenames = ["a", "b", "c"]

        super_self = self

        class Observer(kp.Module):
            def configure(self):
                self.count = 0
                self.filenames = []

            def process(self, blob):
                self.count += 1
                self.filenames.append(blob["filename"])
                return blob

            def finish(self):
                assert self.count == len(filenames)
                super_self.assertListEqual(filenames, self.filenames)

        pipe = kp.Pipeline()
        pipe.attach(FilePump, filenames=filenames)
        pipe.attach(Observer)
        pipe.drain()
