# Filename: test_time.py
# pylint: disable=locally-disabled,C0111,R0904,C0103

import km3pipe as kp
from km3modules.common import (Siphon, Delete, Keep, Wrap, Dump, StatusBar,
                               TickTock, MemoryObserver)
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


class TestWrap(TestCase):
    def test_wrap_a_single_value(self):
        def add_a_keyval_dict(blob):
            blob['a'] = {'b': 1, 'c': 2.3}
            return blob

        def check_wrapped_key(blob):
            assert 'Generic Table' == blob['a'].name
            assert 1 == len(blob['a'])
            assert 1 == blob['a'].b
            assert 2.3 == blob['a'].c
            return blob

        pipe = kp.Pipeline()
        pipe.attach(InfinitePump)
        pipe.attach(add_a_keyval_dict)
        pipe.attach(Wrap, key='a')
        pipe.attach(check_wrapped_key)
        pipe.drain(3)

    def test_wrapping_none_is_skipped(self):
        def add_a_keyval_none(blob):
            blob['a'] = None
            return blob

        def check_wrapped_key(blob):
            assert blob['a'] is None
            return blob

        pipe = kp.Pipeline()
        pipe.attach(InfinitePump)
        pipe.attach(add_a_keyval_none)
        pipe.attach(Wrap, key='a')
        pipe.attach(check_wrapped_key)
        pipe.drain(3)

    def test_wrap_multiple_values(self):
        def add_a_keyval_dict(blob):
            blob['a'] = {'b': 1, 'c': 2.3}
            blob['d'] = {'e': [4, 5], 'f': [6.7, 8.9]}
            return blob

        def check_wrapped_key(blob):
            assert 'Generic Table' == blob['a'].name
            assert 1 == len(blob['a'])
            assert 1 == blob['a'].b
            assert 2.3 == blob['a'].c
            assert 'Generic Table' == blob['d'].name
            assert 2 == len(blob['d'])
            assert 4 == blob['d'].e[0]
            assert 8.9 == blob['d'].f[1]
            return blob

        pipe = kp.Pipeline()
        pipe.attach(InfinitePump)
        pipe.attach(add_a_keyval_dict)
        pipe.attach(Wrap, keys=['a', 'd'])
        pipe.attach(check_wrapped_key)
        pipe.drain(3)


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
