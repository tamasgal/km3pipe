__author__ = 'tamasgal'

import unittest

from km3pipe.core import Pipeline, Module, Blob


class TestPipeline(unittest.TestCase):
    """Tests for the main pipeline"""

    def test_attach(self):
        pl = Pipeline()
        pl.attach(1)
        pl.attach(2)
        self.assertListEqual([1, 2], pl.modules)

    def test_drain(self):
        pl = Pipeline()
        pl.drain()


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


class TestBlob(unittest.TestCase):
    """Tests for the blob holding the data"""

    def test_init(self):
        blob = Blob()