from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

import unittest
from mock import MagicMock

from km3pipe.core import Pipeline, Module, Blob


class TestPipeline(unittest.TestCase):
    """Tests for the main pipeline"""

    def test_attach(self):
        pl = Pipeline()
        pl.attach(Module, 'module1')
        pl.attach(Module, 'module2')
        self.assertEqual('module1', pl.modules[0].name)
        self.assertEqual('module2', pl.modules[1].name)

    def test_drain_calls_process_method_on_each_attached_module(self):
        pl = Pipeline(blob=1, cycles=1)
        pl.attach(Module, 'module1')
        pl.attach(Module, 'module2')
        for module in pl.modules:
            module.process = MagicMock(return_value=1)
        pl.drain()
        for module in pl.modules:
            module.process.assert_called_once_with(1)

    def test_finish(self):
        pl = Pipeline()
        pl.finish()

    def test_drain_calls_finish_on_each_attached_module(self):
        pl = Pipeline(cycles=4)
        pl.attach(Module, 'module1')
        pl.attach(Module, 'module2')
        for module in pl.modules:
            module.finish = MagicMock()
        pl.drain()
        for module in pl.modules:
            module.finish.assert_called_once_with()


class TestModule(unittest.TestCase):
    """Tests for the pipeline module"""

    def test_name_can_be_set_on_init(self):
        name = 'foo'
        module = Module(name=name)
        self.assertEqual(name, module.name)

    def test_name_is_read_only(self):
        module = Module(name='foo')
        with self.assertRaises(AttributeError):
            module.name = 'narf'

    def test_process(self):
        blob = Blob()
        module = Module(name='foo')
        processed_blob = module.process(blob)
        self.assertIs(blob, processed_blob)

    def test_add_parameter(self):
        module = Module()
        module.add('foo', 'default', 'help')
        self.assertDictEqual({'foo': 'default'}, module.parameters)

    def test_get_parameter(self):
        module = Module()
        module.add('foo', 'default', 'help')
        self.assertEqual('default', module.get('foo'))

    def test_default_parameter_value_can_be_overwritten(self):
        class Foo(Module):
            def __init__(self, **context):
                super(self.__class__, self).__init__(**context)
                self.foo = self.get('foo') or 'default_foo'
        module = Foo()
        self.assertEqual('default_foo', module.foo)
        module = Foo(foo='overwritten')
        self.assertEqual('overwritten', module.foo)

    def test_finish(self):
        module = Module()
        module.finish()

class TestBlob(unittest.TestCase):
    """Tests for the blob holding the data"""

    def test_init(self):
        blob = Blob()

    def test_field_can_be_added(self):
        blob = Blob()
        blob['foo'] = 1
        self.assertEqual(1, blob['foo'])