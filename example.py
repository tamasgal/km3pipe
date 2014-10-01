from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

from km3pipe.core import Pipeline, Module


class Pump(Module):

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.blobs = self.get('blobs') or [{'foo': 1}, {'bar': 2}]

    def process(self, blob):
        try:
            return self.blobs.pop(0)
        except IndexError:
            raise StopIteration


class Foo(Module):

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.foo = self.get('foo') or 'default_foo'
        self.bar = self.get('bar') or 23

    def process(self, blob):
        print("This is the current blob: " + str(blob))
        print("foo={0}".format(self.foo))
        print("bar={0}".format(self.bar))
        blob['foo'] = self.foo
        return blob

class Moo(Module):
    def process(self, blob):
        #blob['moo_entry'] = 42
        print("This is the blob after processing Moo: " + str(blob))
        return blob


pipe = Pipeline()
pipe.attach(Pump, 'the_pump')
pipe.attach(Foo, 'foo_module', foo='dummyfoo', bar='dummybar')
pipe.attach(Moo, 'moo_module')
pipe.drain()
