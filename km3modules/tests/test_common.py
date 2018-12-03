# Filename: test_time.py
# pylint: disable=locally-disabled,C0111,R0904,C0103

import km3pipe as kp
from km3pipe.dataclasses import Table
from km3modules.common import (
    Siphon, Delete, Keep, Dump, StatusBar, TickTock, MemoryObserver,
    BlobIndexer
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


class InfinitePump(kp.Pump):
    """A pump which just infinetly spits out indexed blobs"""

    def configure(self):
        self.i = 0

    def process(self, blob):
        self.i += 1
        blob['i'] = self.i
        return blob


class TestKeep(TestCase):
    def test_keep_a_single_key(self):
        class APump(kp.Pump):
            def process(self, blob):
                blob['a'] = 'a'
                blob['b'] = 'b'
                blob['c'] = 'c'
                blob['d'] = 'd'
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert 'a' not in blob
                assert 'b' not in blob
                assert 'c' not in blob
                assert 'd' == blob['d']
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Keep, keys='d')
        pipe.attach(Observer)
        pipe.drain(5)

    def test_keep_multiple_keys(self):
        class APump(kp.Pump):
            def process(self, blob):
                blob['a'] = 'a'
                blob['b'] = 'b'
                blob['c'] = 'c'
                blob['d'] = 'd'
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert 'a' not in blob
                assert 'b' == blob['b']
                assert 'c' not in blob
                assert 'd' == blob['d']
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Keep, keys=['b', 'd'])
        pipe.attach(Observer)
        pipe.drain(5)

    def test_hdf5_keep_group_wo_subgroup(self):
        class APump(kp.Pump):
            def process(self, blob):
                blob['A'] = kp.Table({
                    'foo': [1, 2, 3],
                    'bar': [4, 5, 6]
                },
                                     h5loc='/foobar')
                blob['B'] = kp.Table({
                    'a': [1.1, 2.1, 3.1],
                    'b': [4.2, 5.2, 6.2]
                },
                                     h5loc='/ab')
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert 'A' in blob.keys()
                assert '/foobar' == blob['A'].h5loc
                assert not 'B' in blob.keys()
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Keep, h5locs=['/foobar'])
        pipe.attach(Observer)
        pipe.drain(5)

    def test_hdf5_keep_group_w_subgroup(self):
        class APump(kp.Pump):
            def process(self, blob):
                blob['A'] = kp.Table({
                    'foo': [1, 2, 3],
                    'bar': [4, 5, 6]
                },
                                     h5loc='/foobar')
                blob['B'] = kp.Table({
                    'a': [1.1, 2.1, 3.1],
                    'b': [4.2, 5.2, 6.2]
                },
                                     h5loc='/ab')
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert 'A' in blob.keys()
                assert '/foobar' == blob['A'].h5loc
                assert not 'B' in blob.keys()
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Keep, h5locs=['/foobar'])
        pipe.attach(Observer)
        pipe.drain(5)

    def test_key_hdf5_group_individual(self):
        class APump(kp.Pump):
            def process(self, blob):
                blob['A'] = kp.Table({
                    'foo': [1, 2, 3],
                    'bar': [4, 5, 6]
                },
                                     h5loc='/foobar')
                blob['B'] = kp.Table({
                    'a': [1.1, 2.1, 3.1],
                    'b': [4.2, 5.2, 6.2]
                },
                                     h5loc='/ab')
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert 'A' in blob.keys()
                assert '/foobar' == blob['A'].h5loc
                assert 'B' in blob.keys()
                assert '/ab' == blob['B'].h5loc
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Keep, keys=['B'], h5locs=['/foobar'])
        pipe.attach(Observer)
        pipe.drain(5)

    def test_key_hdf5_group_parallel(self):
        class APump(kp.Pump):
            def process(self, blob):
                blob['A'] = kp.Table({
                    'foo': [1, 2, 3],
                    'bar': [4, 5, 6]
                },
                                     h5loc='/foobar')
                blob['B'] = kp.Table({
                    'a': [1.1, 2.1, 3.1],
                    'b': [4.2, 5.2, 6.2]
                },
                                     h5loc='/ab')
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert 'A' in blob.keys()
                assert '/foobar' == blob['A'].h5loc
                assert not 'B' in blob.keys()
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Keep, keys=['A'], h5locs=['/foobar'])
        pipe.attach(Observer)
        pipe.drain(5)

    def test_major_hdf5_group(self):
        class APump(kp.Pump):
            def process(self, blob):
                blob['A'] = kp.Table({
                    'foo': [1, 2, 3],
                    'bar': [4, 5, 6]
                },
                                     h5loc='/foobar/a')
                blob['B'] = kp.Table({
                    'a': [1.1, 2.1, 3.1],
                    'b': [4.2, 5.2, 6.2]
                },
                                     h5loc='/foobar/b')
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert 'A' in blob.keys()
                assert '/foobar/a' == blob['A'].h5loc
                assert 'B' in blob.keys()
                assert '/foobar/b' == blob['B'].h5loc
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Keep, h5locs=['/foobar'])
        pipe.attach(Observer)
        pipe.drain(5)


class TestDelete(TestCase):
    def test_delete_a_single_key(self):
        class APump(kp.Pump):
            def process(self, blob):
                blob['a'] = 'a'
                blob['b'] = 'b'
                blob['c'] = 'c'
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert 'a' == blob['a']
                assert 'b' not in blob
                assert 'c' == blob['c']
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Delete, key='b')
        pipe.attach(Observer)
        pipe.drain(5)

    def test_delete_multiple_keys(self):
        class APump(kp.Pump):
            def process(self, blob):
                blob['a'] = 'a'
                blob['b'] = 'b'
                blob['c'] = 'c'
                return blob

        class Observer(kp.Module):
            def process(self, blob):
                assert 'a' not in blob
                assert 'b' not in blob
                assert 'c' == blob['c']
                return blob

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Delete, keys=['a', 'b'])
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
            blob['a'] = 1
            return blob

        pipe = kp.Pipeline()
        pipe.attach(InfinitePump)
        pipe.attach(add_something)
        pipe.attach(Dump)
        pipe.drain(3)

    def test_dump_a_key(self):
        def add_something(blob):
            blob['a'] = 1
            return blob

        pipe = kp.Pipeline()
        pipe.attach(InfinitePump)
        pipe.attach(add_something)
        pipe.attach(Dump, key='a')
        pipe.drain(3)

    def test_dump_multiple_keys(self):
        def add_something(blob):
            blob['a'] = 1
            blob['b'] = 2
            return blob

        pipe = kp.Pipeline()
        pipe.attach(InfinitePump)
        pipe.attach(add_something)
        pipe.attach(Dump, keys=['a', 'b'])
        pipe.drain(3)

    def test_dump_full(self):
        def add_something(blob):
            blob['a'] = 1
            blob['b'] = 2
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
                assert blob['blob_index'] == self.index
                self.index += 1
                return blob

        pipe = kp.Pipeline()
        pipe.attach(InfinitePump)
        pipe.attach(BlobIndexer)
        pipe.attach(Observer)
        pipe.drain(4)
