# coding=utf-8
# Filename: test_tools.py
# pylint: disable=locally-disabled,C0111,R0904,C0103
from __future__ import division, absolute_import, print_function

import numpy as np
import itertools
from datetime import datetime, timedelta
from time import sleep

from km3pipe.testing import TestCase, MagicMock, StringIO
from km3pipe.tools import (unpack_nfirst, split, namedtuple_with_defaults,
                           geant2pdg, pdg2name, Cuckoo, total_seconds,
                           remain_file_pointer, decamelise, camelise, Timer)

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


    def test_geant2pdg(self):
        self.assertEqual(22, geant2pdg(1))
        self.assertEqual(-13, geant2pdg(5))

    def test_geant2pdg_returns_0_for_unknown_particle_id(self):
        self.assertEqual(0, geant2pdg(-999))

    def test_pdg2name(self):
        self.assertEqual('mu-', pdg2name(13))
        self.assertEqual('anu_tau', pdg2name(-16))

    def test_pdg2name_returns_NA_for_unknown_particle(self):
        self.assertEqual('N/A', pdg2name(0))

    def test_total_seconds(self):
        seconds = 3
        time1 = datetime.now()
        time2 = time1 + timedelta(seconds=seconds)
        td = time2 - time1
        self.assertAlmostEqual(seconds, total_seconds(td))


class TestCuckoo(TestCase):
    def test_reset_timestamp(self):
        cuckoo = Cuckoo()
        cuckoo.reset()
        delta = datetime.now() - cuckoo.timestamp
        self.assertGreater(total_seconds(delta), 0)

    def test_set_interval_on_init(self):
        cuckoo = Cuckoo(1)
        self.assertEqual(1, cuckoo.interval)

    def test_set_callback(self):
        callback = 1
        cuckoo = Cuckoo(callback=callback)
        self.assertEqual(1, cuckoo.callback)

    def test_msg_calls_callback(self):
        callback = MagicMock()
        message = 'a'
        cuckoo = Cuckoo(callback=callback)
        cuckoo.msg(message)
        callback.assert_called_with(message)

    def test_msg_calls_callback_with_empty_args(self):
        callback = MagicMock()
        message = 'a'
        cuckoo = Cuckoo(callback=callback)
        cuckoo.msg()
        callback.assert_called_with()

    def test_msg_calls_callback_with_multiple_args(self):
        callback = MagicMock()
        message = 'a'
        cuckoo = Cuckoo(callback=callback)
        cuckoo.msg(1, 2, 3)
        callback.assert_called_with(1, 2, 3)

    def test_msg_calls_callback_with_multiple_kwargs(self):
        callback = MagicMock()
        message = 'a'
        cuckoo = Cuckoo(callback=callback)
        cuckoo.msg(a=1, b=2)
        callback.assert_called_with(a=1, b=2)

    def test_msg_calls_callback_with_mixed_args_and_kwargs(self):
        callback = MagicMock()
        message = 'a'
        cuckoo = Cuckoo(callback=callback)
        cuckoo.msg(1, 2, c=3, d=4)
        callback.assert_called_with(1, 2, c=3, d=4)

    def test_direct_call_calls_callback(self):
        callback = MagicMock()
        message = 'a'
        cuckoo = Cuckoo(callback=callback)
        cuckoo(message)
        callback.assert_called_with(message)

    def test_msg_is_not_called_when_interval_not_reached(self):
        callback = MagicMock()
        message = 'a'
        cuckoo = Cuckoo(10, callback)
        cuckoo.reset()
        cuckoo.msg(message)
        self.assertFalse(callback.called)

    def test_msg_is_only_called_when_interval_reached(self):
        callback = MagicMock()
        message = 'a'
        cuckoo = Cuckoo(0.1, callback)
        cuckoo.reset()
        cuckoo.msg(message)
        self.assertFalse(callback.called)
        sleep(0.11)
        cuckoo.msg(message)
        self.assertTrue(callback.called)

    def test_msg_sets_timestamp_on_first_call(self):
        cuckoo = Cuckoo()
        cuckoo.msg()
        assert cuckoo.timestamp

    def test_msg_gets_called_on_the_very_first_time(self):
        callback = MagicMock()
        message = 'a'
        cuckoo = Cuckoo(1, callback)
        cuckoo.msg(message)
        self.assertTrue(callback.called)

    def test_msg_resets_timestamp_after_interval_reached(self):
        callback = MagicMock()
        message = 'a'
        cuckoo = Cuckoo(0.1, callback)
        cuckoo.reset()
        timestamp1 = cuckoo.timestamp
        print(cuckoo.timestamp)
        self.assertFalse(callback.called)
        sleep(0.11)
        cuckoo.msg(message)
        timestamp2 = cuckoo.timestamp
        print(cuckoo.timestamp)
        self.assertTrue(callback.called)
        assert timestamp1 is not timestamp2

    def test_interval_reached(self):
        cuckoo = Cuckoo(0.1)
        cuckoo.reset()
        self.assertFalse(cuckoo._interval_reached())
        sleep(0.11)
        self.assertTrue(cuckoo._interval_reached())


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


class TestTimer(TestCase):
    def test_init(self):
        t = Timer()

    def test_context_manager(self):
        mock = MagicMock()
        with Timer(callback=mock) as t:
            pass
        mock.assert_called_once()

    def test_context_manager_calls_with_standard_text(self):
        mock = MagicMock()
        with Timer(callback=mock) as t:
            pass
        self.assertTrue(mock.call_args[0][0].startswith("It "))
