# Filename: test_core.py
# pylint: disable=C0111,E1003,R0904,C0103,R0201,C0102
from __future__ import unicode_literals

import tempfile
from io import StringIO

from km3pipe.testing import TestCase, MagicMock
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

    def test_attach_bundle(self):
        modules = [Module, Module]
        self.pl.attach_bundle(modules)
        self.assertEqual(2, len(self.pl.modules))

    def test_attach_function(self):
        self.pl.attach(lambda x: 1)
        self.pl.attach(lambda x: 2, "Another Lambda")
        self.assertEqual('<lambda>', self.pl.modules[0].name)
        self.assertEqual('Another Lambda', self.pl.modules[1].name)

    def test_drain_calls_each_attached_module(self):
        pl = Pipeline(blob=1)

        func_module_spy = MagicMock()

        def func_module(blob):
            func_module_spy()
            return blob

        pl.attach(Module, 'module1')
        pl.attach(func_module, 'module2')
        pl.attach(Module, 'module3')

        for module in pl.modules:
            print(module)
            if isinstance(module, Module):
                print("normal module, mocking")
                module.process = MagicMock(return_value={})

        n = 3

        pl.drain(n)

        for module in pl.modules:
            try:
                # Regular modules
                self.assertEqual(n, module.process.call_count)
            except AttributeError:
                # Function module
                self.assertEqual(n, func_module_spy.call_count)

    def test_drain_calls_process_method_on_each_attached_module(self):
        pl = Pipeline(blob=1)

        pl.attach(Module, 'module1')
        pl.attach(Module, 'module2')
        pl.attach(Module, 'module3')
        for module in pl.modules:
            module.process = MagicMock(return_value={})
        n = 3
        pl.drain(n)
        for module in pl.modules:
            self.assertEqual(n, module.process.call_count)

    def test_drain_doesnt_call_process_if_blob_is_none(self):
        pl = Pipeline(blob=1)

        pl.attach(Module, 'module1')
        pl.attach(Module, 'module2')
        pl.attach(Module, 'module3')
        pl.modules[0].process = MagicMock(return_value=None)
        pl.modules[1].process = MagicMock(return_value={})
        pl.modules[2].process = MagicMock(return_value={})
        n = 3
        pl.drain(n)
        self.assertEqual(n, pl.modules[0].process.call_count)
        self.assertEqual(0, pl.modules[1].process.call_count)
        self.assertEqual(0, pl.modules[2].process.call_count)

    def test_conditional_module_not_called_if_key_not_in_blob(self):
        pl = Pipeline(blob=1)

        pl.attach(Module, 'module1')
        pl.attach(Module, 'module2', only_if='foo')
        pl.attach(Module, 'module3')

        for module in pl.modules:
            module.process = MagicMock(return_value={})

        pl.drain(1)

        self.assertEqual(1, pl.modules[0].process.call_count)
        self.assertEqual(0, pl.modules[1].process.call_count)
        self.assertEqual(1, pl.modules[2].process.call_count)

    def test_conditional_module_called_if_key_in_blob(self):
        pl = Pipeline(blob=1)

        pl.attach(Module, 'module1')
        pl.attach(Module, 'module2', only_if='foo')
        pl.attach(Module, 'module3')

        pl.modules[0].process = MagicMock(return_value={'foo': 23})
        pl.modules[1].process = MagicMock(return_value={})
        pl.modules[2].process = MagicMock(return_value={})

        pl.drain(1)

        self.assertEqual(1, pl.modules[0].process.call_count)
        self.assertEqual(1, pl.modules[1].process.call_count)
        self.assertEqual(1, pl.modules[2].process.call_count)

    def test_condition_every(self):
        pl = Pipeline(blob=1)

        pl.attach(Module, 'module1')
        pl.attach(Module, 'module2', every=3)
        pl.attach(Module, 'module3', every=9)
        pl.attach(Module, 'module4', every=10)
        pl.attach(Module, 'module5')

        func_module = MagicMock()
        func_module.__name__ = "MagicMock"
        pl.attach(func_module, 'funcmodule', every=4)

        for module in pl.modules:
            module.process = MagicMock(return_value={})

        pl.drain(9)

        self.assertEqual(9, pl.modules[0].process.call_count)
        self.assertEqual(3, pl.modules[1].process.call_count)
        self.assertEqual(1, pl.modules[2].process.call_count)
        self.assertEqual(0, pl.modules[3].process.call_count)
        self.assertEqual(9, pl.modules[4].process.call_count)
        self.assertEqual(2, func_module.call_count)

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
        self.assertEqual(1, pl.modules[0].call_count)
        self.assertEqual(1, pl.modules[1].call_count)
        self.assertEqual(1, pl.modules[2].call_count)

    def test_finish(self):
        out = self.pl.finish()
        assert out is not None

    def test_drain_calls_finish_on_each_attached_module(self):
        self.pl.attach(Module, 'module1')
        self.pl.attach(Module, 'module2')
        self.pl.attach(lambda x: 1, 'func_module')
        for module in self.pl.modules:
            module.finish = MagicMock()
        self.pl.drain(4)
        for module in self.pl.modules:
            if module.name != 'func_module':
                self.assertEqual(1, module.finish.call_count)

    def test_ctrl_c_handling(self):
        pl = Pipeline()
        self.assertFalse(pl._stop)
        pl._handle_ctrl_c()    # first KeyboardInterrupt
        self.assertTrue(pl._stop)
        with self.assertRaises(SystemExit):
            pl._handle_ctrl_c()    # second KeyboardInterrupt

    def test_attaching_a_pump_allows_first_param_to_be_passed_as_fname(self):
        class APump(Pump):
            def configure(self):
                self.filename = self.require('filename')

        p = APump('test')
        self.assertEqual('test', p.filename)

    def test_attaching_multiple_pumps(self):
        pl = Pipeline()

        class APump(Pump):
            def configure(self):
                self.i = 0

            def process(self, blob):
                blob['A'] = self.i
                self.i += 1
                return blob

        class BPump(Pump):
            def configure(self):
                self.i = 0

            def process(self, blob):
                blob['B'] = self.i
                self.i += 1
                return blob

        class CheckBlob(Module):
            def configure(self):
                self.i = 0

            def process(self, blob):
                assert self.i == blob['A']
                assert self.i == blob['B']
                self.i += 1
                return blob

        pl.attach(APump)
        pl.attach(BPump)
        pl.attach(CheckBlob)
        pl.drain(5)


class TestPipelineConfigurationViaFile(TestCase):
    """Auto-configuration of pipelines using TOML files"""

    def test_configuration(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fname = str(fobj.name)
        Pipeline(configfile=fname)
        fobj.close()

    def test_configuration_with_config_for_a_module(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fobj.write(b"[A]\na = 1")
        fobj.flush()
        fname = str(fobj.name)

        class A(Module):
            def process(self, blob):
                assert 1 == self.a
                return blob

        pipe = Pipeline(configfile=fname)
        pipe.attach(A)
        pipe.drain(1)

        fobj.close()

    def test_configuration_with_config_for_multiple_modules(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fobj.write(b"[A]\na = 1\nb = 2\n[B]\nc='d'")
        fobj.flush()
        fname = str(fobj.name)

        class A(Module):
            def process(self, blob):
                assert 1 == self.a
                assert 2 == self.b
                return blob

        class B(Module):
            def process(self, blob):
                assert 'd' == self.c
                return blob

        pipe = Pipeline(configfile=fname)
        pipe.attach(A)
        pipe.attach(B)
        pipe.drain(1)

        fobj.close()

    def test_configuration_with_named_modules(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fobj.write(b"[X]\na = 1\nb = 2\n[Y]\nc='d'")
        fobj.flush()
        fname = str(fobj.name)

        class A(Module):
            def process(self, blob):
                assert 1 == self.a
                assert 2 == self.b
                return blob

        class B(Module):
            def process(self, blob):
                assert 'd' == self.c
                return blob

        pipe = Pipeline(configfile=fname)
        pipe.attach(A, 'X')
        pipe.attach(B, 'Y')
        pipe.drain(1)

        fobj.close()

    def test_configuration_precedence_over_kwargs(self):
        fobj = tempfile.NamedTemporaryFile(delete=True)
        fobj.write(b"[A]\na = 1\nb = 2")
        fobj.flush()
        fname = str(fobj.name)

        class A(Module):
            def process(self, blob):
                assert 1 == self.a
                assert 2 == self.b
                return blob

        pipe = Pipeline(configfile=fname)
        pipe.attach(A, b='foo')
        pipe.drain(1)

        fobj.close()


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

    def test_context(self):
        with Pump() as p:
            print(p)


class TestBlob(TestCase):
    """Tests for the blob holding the data"""

    def test_field_can_be_added(self):
        blob = Blob()
        blob['foo'] = 1
        self.assertEqual(1, blob['foo'])


class TestServices(TestCase):
    def setUp(self):
        self.pl = Pipeline()

    def test_service(self):
        class Service(Module):
            def configure(self):
                self.expose(23, "foo")
                self.expose(self.whatever, "whatever")

            def whatever(self, x):
                return x * 2

        class UseService(Module):
            def process(self, blob):
                print(self.services)
                assert 23 == self.services["foo"]
                assert 2 == self.services["whatever"](1)

        self.pl.attach(Service)
        self.pl.attach(UseService)
        self.pl.drain(1)

    def test_service_usable_in_configure_when_attached_before(self):
        return

        class Service(Module):
            def configure(self):
                self.expose(23, "foo")
                self.expose(self.whatever, "whatever")

            def whatever(self, x):
                return x * 2

        class UseService(Module):
            def configure(self):
                assert 23 == self.services["foo"]
                assert 2 == self.services["whatever"](1)

        self.pl.attach(Service)
        self.pl.attach(UseService)
        self.pl.drain(1)

    def test_required_service(self):
        class AService(Module):
            def configure(self):
                self.expose(self.a_function, 'a_function')

            def a_function(self, b='c'):
                return b + 'd'

        class AModule(Module):
            def configure(self):
                self.require_service('a_function', why='because')

            def process(self, blob):
                assert 'ed' == self.services['a_function']("e")

        self.pl.attach(AService)
        self.pl.attach(AModule)
        self.pl.drain(2)

    def test_required_service_not_present(self):
        self.pl.log = MagicMock()

        class AModule(Module):
            def configure(self):
                self.require_service('a_function', why='because')

            def process(self, blob):
                assert False    # make sure that process is not called

        self.pl.attach(AModule)
        self.pl.drain(1)

        self.pl.log.critical.assert_called_with(
            'Following services are required and missing: a_function'
        )

    def test_required_service_not_present_in_multiple_modules(self):
        self.pl.log = MagicMock()

        class AModule(Module):
            def configure(self):
                self.require_service('a_function', why='because')
                self.require_service('b_function', why='because')

            def process(self, blob):
                assert False    # make sure that process is not called

        class BModule(Module):
            def configure(self):
                self.require_service('c_function', why='because')

            def process(self, blob):
                assert False    # make sure that process is not called

        self.pl.attach(AModule)
        self.pl.attach(BModule)
        self.pl.drain(1)

        self.pl.log.critical.assert_called_with(
            'Following services are required and missing: '
            'a_function, b_function, c_function'
        )

    def test_required_service_not_present_but_some_are_present(self):
        self.pl.log = MagicMock()

        class AModule(Module):
            def configure(self):
                self.expose(self.d_function, 'd_function')
                self.require_service('a_function', why='because')
                self.require_service('b_function', why='because')

            def d_function(self):
                pass

            def process(self, blob):
                assert False    # make sure that process is not called

        class BModule(Module):
            def configure(self):
                self.require_service('c_function', why='because')

            def process(self, blob):
                assert False    # make sure that process is not called

        class CModule(Module):
            def configure(self):
                self.require_service('d_function', why='because')

            def process(self, blob):
                assert False    # make sure that process is not called

        self.pl.attach(AModule)
        self.pl.attach(BModule)
        self.pl.attach(CModule)
        self.pl.drain(1)

        self.pl.log.critical.assert_called_with(
            'Following services are required and missing: '
            'a_function, b_function, c_function'
        )
