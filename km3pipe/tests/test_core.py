# coding=utf-8
# Filename: test_core.py
# pylint: disable=C0111,E1003,R0904,C0103,R0201,C0102
from __future__ import division, absolute_import, print_function

from km3pipe.testing import TestCase, StringIO, MagicMock
from km3pipe.core import Pipeline, Module, Pump, Blob

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestPipeline(TestCase):
    """Tests for the main pipeline"""

    def setUp(self):
        self.pl = Pipeline()

    def test_attach(self):
        self.pl.attach(Module, 'module1')
        self.pl.attach(Module, 'module2')
        print([m.name for m in self.pl.modules])
        self.assertEqual('module1', self.pl.modules[0].name)
        self.assertEqual('module2', self.pl.modules[1].name)

    def test_attach_function(self):
        self.pl.attach(lambda x: 1)
        self.pl.attach(lambda x: 2, "Another Lambda")
        self.assertEqual('<lambda>', self.pl.modules[0].name)
        self.assertEqual('Another Lambda', self.pl.modules[1].name)

    def test_drain_calls_each_attached_module(self):
        pl = Pipeline(blob=1)

        func_module = MagicMock()
        func_module.__name__ = "MagicMock"

        pl.attach(Module, 'module1')
        pl.attach(func_module, 'module2')
        pl.attach(Module, 'module3')

        for module in pl.modules:
            if isinstance(module, Module):
                module.__call__ = MagicMock(return_value={})

        pl.drain(1)

        for module in pl.modules:
            try:
                # Regular modules
                module.__call__.assert_called_once()
            except AttributeError:
                # Function modules
                module.assert_called_once()

    def test_drain_calls_process_method_on_each_attached_module(self):
        pl = Pipeline(blob=1)

        pl.attach(Module, 'module1')
        pl.attach(Module, 'module2')
        pl.attach(Module, 'module3')
        for module in pl.modules:
            module.process = MagicMock(return_value={})
        pl.drain(1)
        for module in pl.modules:
            module.process.assert_called_once()

    def test_drain_calls_function_modules(self):
        pl = Pipeline(blob=1)

        func_module1 = MagicMock()
        func_module2 = MagicMock()
        func_module3 = MagicMock()

        func_module1.__name__ = "MagicMock"
        func_module2.__name__ = "MagicMock"
        func_module3.__name__ = "MagicMock"

        pl.attach(func_module1, 'module1')
        pl.attach(func_module2, 'module2')
        pl.attach(func_module3, 'module3')
        pl.drain(1)
        for module in pl.modules:
            module.assert_called_once()

    def test_finish(self):
        self.pl.finish()

    def test_drain_calls_finish_on_each_attached_module(self):
        self.pl.attach(Module, 'module1')
        self.pl.attach(Module, 'module2')
        self.pl.attach(lambda x: 1, 'func_module')
        for module in self.pl.modules:
            module.finish = MagicMock()
        self.pl.drain(4)
        for module in self.pl.modules:
            if module.name != 'func_module':
                module.finish.assert_called_once_with()


class TestModule(TestCase):
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
        module.add('foo', 'default')
        self.assertDictEqual({'foo': 'default'}, module.parameters)

    def test_get_parameter(self):
        module = Module()
        module.add('foo', 'default')
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


class TestPump(TestCase):
    """Tests for the pump"""

    def test_rewind_file(self):
        pump = Pump()
        test_file = StringIO("Some content")
        pump.blob_file = test_file
        pump.blob_file.read(1)
        self.assertEqual(1, pump.blob_file.tell())
        pump.rewind_file()
        self.assertEqual(0, pump.blob_file.tell())


class TestBlob(TestCase):
    """Tests for the blob holding the data"""

    def test_field_can_be_added(self):
        blob = Blob()
        blob['foo'] = 1
        self.assertEqual(1, blob['foo'])
