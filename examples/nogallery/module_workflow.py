#!/usr/bin/env python
from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

from km3pipe.core import Pipeline, Module, Pump


class DummyPump(Pump):
    """A pump demonstration with a dummy list as data."""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.data = [{'nr': 1}, {'nr': 2}]
        self.blobs = self.blob_generator()

    def process(self, blob):
        return next(self.blobs)

    def blob_generator(self):
        """Create a blob generator."""
        for blob in self.data:
            yield blob


class Foo(Module):
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.foo = self.get('foo') or 'default_foo'
        self.bar = self.get('bar') or 23

    def process(self, blob):
        print("This is the current blob: " + str(blob))
        blob['foo_entry'] = self.foo
        return blob


class Moo(Module):
    def process(self, blob):
        blob['moo_entry'] = 42
        return blob


class PrintBlob(Module):
    def process(self, blob):
        print(blob)
        return blob


pipe = Pipeline()
pipe.attach(Pump, 'the_pump')
pipe.attach(Foo, 'foo_module', foo='dummyfoo', bar='dummybar')
pipe.attach(Moo, 'moo_module')
pipe.attach(PrintBlob, 'print_blob')
pipe.drain()
