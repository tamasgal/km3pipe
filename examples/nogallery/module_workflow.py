#!/usr/bin/env python

__author__ = 'tamasgal'

from km3pipe.core import Pipeline, Module, Pump


class DummyPump(Pump):
    """A pump demonstration with a dummy list as data."""

    def configure(self):
        self.data = [{'nr': 1}, {'nr': 2}]
        self.blobs = self.blob_generator()

    def process(self, blob):
        return next(self.blobs)

    def blob_generator(self):
        """Create a blob generator."""
        for blob in self.data:
            yield blob


class Foo(Module):
    """A dummy module with optional and required parameters"""

    def configure(self):
        self.foo = self.get('foo', default='default_foo')    # optional
        self.bar = self.get('bar', default=23)    # optional
        self.baz = self.require('baz')    # required
        self.i = 0

    def process(self, blob):
        print("This is the current blob: " + str(blob))
        self.i += 1
        blob['foo_entry'] = self.foo
        return blob

    def finish(self):
        print("My process() method was called {} times.".format(self.i))


def moo(blob):
    """A simple function to attach"""
    blob['moo_entry'] = 42
    return blob


class PrintBlob(Module):
    def process(self, blob):
        print(blob)
        return blob


pipe = Pipeline()
pipe.attach(DummyPump, 'the_pump')
pipe.attach(Foo, bar='dummybar', baz=69)
pipe.attach(moo)
pipe.attach(PrintBlob)
pipe.drain()
