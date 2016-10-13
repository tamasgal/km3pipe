# coding=utf-8
# Filename: test_core.py
# pylint: disable=C0111,E1003,R0904,C0103,R0201,C0102
from __future__ import division, absolute_import, print_function

from km3pipe.testing import TestCase, StringIO, MagicMock
from km3pipe.core import Pipeline, Module, Pump, Blob, Geometry
from km3pipe.dataclasses import HitSeries

import numpy as np

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
                self.assertEqual(1, module.finish.call_count)

    def test_ctrl_c_handling(self):
        pl = Pipeline()
        self.assertFalse(pl._stop)
        pl._handle_ctrl_c()  # first KeyboardInterrupt
        self.assertTrue(pl._stop)
        with self.assertRaises(SystemExit):
            pl._handle_ctrl_c()  # second KeyboardInterrupt


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


class TestGeometry(TestCase):
    """Tests for the Geometry class"""

    def test_apply_to_hitseries(self):

        class FakeDetector(object):
            def pmt_with_id(self, i):
                pmt = MagicMock(dir=np.array((i*10+i, i*10+i+1, i*10+i+2)),
                                pos=np.array((i*100+i, i*100+i+1, i*100+i+2)),
                                t0=1000*i)
                return pmt

        geo = Geometry(detector=FakeDetector())

        n = 5
        ids = np.arange(n)
        dom_ids = np.arange(n)
        times = np.arange(n)
        tots = np.arange(n)
        channel_ids = np.arange(n)
        triggereds = np.ones(n)
        pmt_ids = np.arange(n)

        hits = HitSeries.from_arrays(
            channel_ids, dom_ids,
            ids,
            pmt_ids,
            times,
            tots,
            triggereds,
            0,      # event_id
        )

        self.assertEqual(0, hits[0].time)
        self.assertEqual(4, hits[4].time)
        self.assertTrue(np.isnan(hits[2].pos[1]))

        geo._apply_to_hitseries(hits)

        self.assertAlmostEqual(303, hits[3].pos[0])
        self.assertAlmostEqual(304, hits[3].pos[1])
        self.assertAlmostEqual(305, hits[3].pos[2])
        self.assertAlmostEqual(406, hits[4].pos[2])
        self.assertAlmostEqual(2, hits[0].dir[2])
        self.assertAlmostEqual(12, hits[1].dir[1])
        self.assertAlmostEqual(22, hits[2].dir[0])

        self.assertEqual(1001, hits[1].time)
        self.assertEqual(4004, hits[4].time)

        for idx, hit in enumerate(hits):
            h = hit
            if idx == 3:
                break

        self.assertAlmostEqual(303, h.pos[0])
