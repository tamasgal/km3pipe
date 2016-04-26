# coding=utf-8
# Filename: test_tools.py
# pylint: disable=locally-disabled,C0111,R0904,C0103
from __future__ import division, absolute_import, print_function

import numpy as np
import itertools
from datetime import datetime
from time import sleep

from km3pipe.testing import TestCase, MagicMock, StringIO
from km3pipe.tools import (unpack_nfirst, split, namedtuple_with_defaults,
                           angle_between, geant2pdg, pdg2name, PMTReplugger,
                           Cuckoo, total_seconds, remain_file_pointer)


class TestTools(TestCase):

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

    def test_angle_between(self):
        v1 = (1, 0, 0)
        v2 = (0, 1, 0)
        v3 = (-1, 0, 0)
        self.assertAlmostEqual(0, angle_between(v1, v1))
        self.assertAlmostEqual(np.pi/2, angle_between(v1, v2))
        self.assertAlmostEqual(np.pi, angle_between(v1, v3))

    def test_angle_between_returns_nan_for_zero_length_vectors(self):
        v1 = (0, 0, 0)
        v2 = (1, 0, 0)
        self.assertTrue(np.isnan(angle_between(v1, v2)))

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
        time1 = datetime.now()
        sleep(0.1)
        time2 = datetime.now()
        td = time2 - time1
        self.assertAlmostEqual(0.1, total_seconds(td), 1)


# [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
PMT_COMBS = list(itertools.combinations(range(4), 2))
ANGLES = range(len(PMT_COMBS))


class TestPMTReplugger(TestCase):

    def setUp(self):
        self.replugger = PMTReplugger(PMT_COMBS, ANGLES, [])

    def test_angle_for(self):
        # self.assertEqual(0, self.replugger.angle_for((0, 1)))
        # self.assertEqual(1, self.replugger.angle_for((0, 2)))
        pass

    def test_switch(self):
        self.replugger.switch([0, 1], [1, 0])
        self.assertEqual(self.replugger._new_combs,
                         [(0, 1), (1, 2), (1, 3), (0, 2), (0, 3), (2, 3)])

    def test_switch_three_indicies(self):
        self.replugger.switch([0, 1, 2], [1, 2, 0])
        self.assertEqual(self.replugger._new_combs,
                         [(1, 2), (0, 1), (1, 3), (0, 2), (2, 3), (0, 3)])

    def test_angle_is_correct_if_two_pmts_are_switched(self):
        self.replugger.switch([0, 1], [1, 0])
        self.assertEqual(0, self.replugger.angle_for((0, 1)))
        self.assertEqual(3, self.replugger.angle_for((0, 2)))
        self.assertEqual(4, self.replugger.angle_for((0, 3)))

    def test_angles_are_ordered_correctly_after_switch(self):
        self.replugger.switch([0, 1, 2], [1, 2, 0])
        self.assertListEqual([1, 3, 5, 0, 2, 4], self.replugger.angles)


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
