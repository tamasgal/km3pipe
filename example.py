from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

from km3pipe.core import Pipeline, Module


class Foo(Module):

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.foo = self.get('foo') or 'default_foo'
        self.bar = self.get('bar') or 23

    def process(self, blob):
        print("Processing " + str(self.name))
        print("This is the current blob: " + str(blob))
        print("foo={0}".format(self.foo))
        print("bar={0}".format(self.bar))
        blob['foo_entry'] = self.foo
        return blob

class Moo(Module):
    def process(self, blob):
        print("Processing " + self.name)
        blob['moo_entry'] = 42
        print("This is the blob after processing Moo: " + str(blob))
        return blob


pipe = Pipeline()
pipe.attach(Foo, 'foo_module')
pipe.attach(Moo, 'moo_module')
pipe.drain()
