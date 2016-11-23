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
                           angle_between, pld3, com, geant2pdg, pdg2name,
                           PMTReplugger, Cuckoo, total_seconds, zenith,
                           azimuth,
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


    def test_zenith(self):
        self.assertAlmostEqual(np.pi, zenith((0, 0, 1)))
        self.assertAlmostEqual(0, zenith((0, 0, -1)))
        self.assertAlmostEqual(np.pi / 2, zenith((0, 1, 0)))
        self.assertAlmostEqual(np.pi / 2, zenith((0, -1, 0)))
        self.assertAlmostEqual(np.pi / 4 * 3, zenith((0, 1, 1)))
        self.assertAlmostEqual(np.pi / 4 * 3, zenith((0, -1, 1)))
        self.assertAlmostEqual(zenith(self.v), 2.5010703409103687)
        self.assertTrue(np.allclose(zenith(self.vecs),
                                    np.array([2.94419709, 2.80175574, 2.50107034,
                                              2.13473897, 1.80873745])))

    def test_azimuth(self):
        self.assertAlmostEqual(0, azimuth((1, 0, 0)))
        self.assertAlmostEqual(np.pi, azimuth((-1, 0, 0)))
        self.assertAlmostEqual(np.pi / 2, azimuth((0, 1, 0)))
        self.assertAlmostEqual(np.pi / 2 * 3, azimuth((0, -1, 0)))
        self.assertAlmostEqual(np.pi / 2 * 3, azimuth((0, -1, 0)))
        self.assertAlmostEqual(0, azimuth((0, 0, 0)))
        self.assertAlmostEqual(azimuth(self.v), 1.10714872)
        self.assertTrue(np.allclose(azimuth(self.vecs),
                                    np.array([1.57079633, 0.78539816, 0.46364761,
                                              0.32175055, 0.24497866])))

    def test_angle_between(self):
        v1 = (1, 0, 0)
        v2 = (0, 1, 0)
        v3 = (-1, 0, 0)
        self.assertAlmostEqual(0, angle_between(v1, v1))
        self.assertAlmostEqual(np.pi/2, angle_between(v1, v2))
        self.assertAlmostEqual(np.pi, angle_between(v1, v3))
        self.assertAlmostEqual(angle_between(self.v, v1), 1.3002465638163236)
        self.assertAlmostEqual(angle_between(self.v, v2), 1.0068536854342678)
        self.assertAlmostEqual(angle_between(self.v, v3), 1.8413460897734695)
        self.assertTrue(np.allclose(angle_between(self.vecs, v1),
                                    np.array([1.57079633, 1.3328552 , 1.00685369,
                                              0.64052231, 0.33983691])))
        self.assertTrue(np.allclose(angle_between(self.vecs, v2),
                                    np.array([1.37340077, 1.3328552 , 1.30024656,
                                              1.30024656, 1.3328552 ])))
        self.assertTrue(np.allclose(angle_between(self.vecs, v3),
                                    np.array([1.57079633, 1.80873745,
                                              2.13473897, 2.50107034,
                                              2.80175574])))


    def test_angle_between_returns_nan_for_zero_length_vectors(self):
        v1 = (0, 0, 0)
        v2 = (1, 0, 0)
        self.assertTrue(np.isnan(angle_between(v1, v2)))

    def test_pld3(self):
        p1 = np.array((0, 0, 0))
        p2 = np.array((0, 0, 1))
        d2 = np.array((0, 1, 0))
        self.assertAlmostEqual(1, pld3(p1, p2, d2))
        p1 = np.array((0, 0, 0))
        p2 = np.array((0, 0, 2))
        d2 = np.array((0, 1, 0))
        self.assertAlmostEqual(2, pld3(p1, p2, d2))
        p1 = np.array((0, 0, 0))
        p2 = np.array((0, 0, 0))
        d2 = np.array((0, 1, 0))
        self.assertAlmostEqual(0, pld3(p1, p2, d2))
        p1 = np.array((1, 2, 3))
        p2 = np.array((4, 5, 6))
        d2 = np.array((7, 8, 9))
        self.assertAlmostEqual(0.5275893, pld3(p1, p2, d2))
        p1 = np.array((0, 0, 2))
        p2 = np.array((-100, 0, -100))
        d2 = np.array((1, 0, 1))
        self.assertAlmostEqual(1.4142136, pld3(p1, p2, d2))
        p1 = np.array([183., -311., 351.96083871])
        p2 = np.array([40.256, -639.888, 921.93])
        d2 = np.array([0.185998, 0.476123, -0.859483])
        self.assertAlmostEqual(21.25456308, pld3(p1, p2, d2))

    def test_com(self):
        center_of_mass = com(((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
        self.assertEqual((5.5, 6.5, 7.5), tuple(center_of_mass))
        center_of_mass = com(((1, 2, 3), (4, 5, 6), (7, 8, 9)),
                             masses=(1, 0, 0))
        self.assertEqual((1, 2, 3), tuple(center_of_mass))
        center_of_mass = com(((1, 1, 1), (0, 0, 0)))
        self.assertEqual((0.5, 0.5, 0.5), tuple(center_of_mass))


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

