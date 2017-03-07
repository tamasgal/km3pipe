# coding=utf-8
# Filename: test_dev.py
# pylint: disable=locally-disabled,C0111,R0904,C0103
from __future__ import division, absolute_import, print_function

import numpy as np

from km3pipe.testing import TestCase, StringIO
from km3pipe.dev import (unpack_nfirst, split, namedtuple_with_defaults,
                           remain_file_pointer, decamelise, camelise)

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestTools(TestCase):
    def setUp(self):
        self.vecs = np.array([[0., 1., 5.],
                              [1., 1., 4.],
                              [2., 1., 3.],
                              [3., 1., 2.],
                              [4., 1., 1.]])
        self.v = (1, 2, 3)
        self.unit_v = np.array([ 0.26726124,  0.53452248,  0.80178373])
        self.unit_vecs = np.array([[ 0.        ,  0.19611614,  0.98058068],
                                  [ 0.23570226,  0.23570226,  0.94280904],
                                  [ 0.53452248,  0.26726124,  0.80178373],
                                  [ 0.80178373,  0.26726124,  0.53452248],
                                  [ 0.94280904,  0.23570226,  0.23570226]])

    def test_unpack_nfirst(self):
        a_tuple = (1, 2, 3, 4, 5)
        a, b, c, rest = unpack_nfirst(a_tuple, 3)
        self.assertEqual(1, a)
        self.assertEqual(2, b)
        self.assertEqual(3, c)
        self.assertTupleEqual((4, 5), rest)

    def test_split_splits_strings(self):
        string = "1 2 3 4"
        parts = split(string)
        self.assertListEqual(['1', '2', '3', '4'], parts)

    def test_split_splits_strings_with_separator(self):
        string = "1,2,3,4"
        parts = split(string, sep=',')
        self.assertListEqual(['1', '2', '3', '4'], parts)

    def test_split_callback_converts_correctly(self):
        string = "1 2 3 4"
        parts = split(string, int)
        self.assertListEqual([1, 2, 3, 4], parts)

        string = "1.0 2.1 3.2 4.3"
        parts = split(string, float)
        self.assertListEqual([1.0, 2.1, 3.2, 4.3], parts)

    def test_namedtuple_with_defaults_initialises_with_none(self):
        Node = namedtuple_with_defaults('Node', 'val left right')
        node = Node()
        self.assertIsNone(node.val)
        self.assertIsNone(node.left)
        self.assertIsNone(node.right)

    def test_namedtuple_with_defaults_initialises_with_given_values(self):
        Node = namedtuple_with_defaults('Node', 'val left right', [1, 2, 3])
        node = Node()
        self.assertEqual(1, node.val)
        self.assertEqual(2, node.left)
        self.assertEqual(3, node.right)


class TestRemainFilePointer(TestCase):

    def test_remains_file_pointer_in_function(self):
        dummy_file = StringIO('abcdefg')

        @remain_file_pointer
        def seek_into_file(file_obj):
            file_obj.seek(1, 0)

        dummy_file.seek(2, 0)
        self.assertEqual(2, dummy_file.tell())
        seek_into_file(dummy_file)
        self.assertEqual(2, dummy_file.tell())

    def test_remains_file_pointer_and_return_value_in_function(self):
        dummy_file = StringIO('abcdefg')

        @remain_file_pointer
        def seek_into_file(file_obj):
            file_obj.seek(1, 0)
            return 1

        dummy_file.seek(2, 0)
        self.assertEqual(2, dummy_file.tell())
        return_value = seek_into_file(dummy_file)
        self.assertEqual(2, dummy_file.tell())
        self.assertEqual(1, return_value)

    def test_remains_file_pointer_in_class_method(self):

        class FileSeekerClass(object):
            def __init__(self):
                self.dummy_file = StringIO('abcdefg')

            @remain_file_pointer
            def seek_into_file(self, file_obj):
                file_obj.seek(1, 0)

        fileseeker = FileSeekerClass()
        fileseeker.dummy_file.seek(2, 0)
        self.assertEqual(2, fileseeker.dummy_file.tell())
        fileseeker.seek_into_file(fileseeker.dummy_file)
        self.assertEqual(2, fileseeker.dummy_file.tell())

    def test_remains_file_pointer_and_return_value_in_class_method(self):

        class FileSeekerClass(object):
            def __init__(self):
                self.dummy_file = StringIO('abcdefg')

            @remain_file_pointer
            def seek_into_file(self, file_obj):
                file_obj.seek(1, 0)
                return 1

        fileseeker = FileSeekerClass()
        fileseeker.dummy_file.seek(2, 0)
        self.assertEqual(2, fileseeker.dummy_file.tell())
        return_value = fileseeker.seek_into_file(fileseeker.dummy_file)
        self.assertEqual(2, fileseeker.dummy_file.tell())
        self.assertEqual(1, return_value)


class TestCamelCaseConverter(TestCase):
    def test_decamelise(self):
        text = "TestCase"
        self.assertEqual("test_case", decamelise(text))
        text = "TestCaseXYZ"
        self.assertEqual("test_case_xyz", decamelise(text))
        text = "1TestCase"
        self.assertEqual("1_test_case", decamelise(text))
        text = "test_case"
        self.assertEqual("test_case", decamelise(text))

    def test_camelise(self):
        text = "camel_case"
        self.assertEqual("CamelCase", camelise(text))
        text = "camel_case"
        self.assertEqual("camelCase", camelise(text, capital_first=False))
