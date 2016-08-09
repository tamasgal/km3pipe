# coding=utf-8
# Filename: test_dataclasses.py
# pylint: disable=C0111,R0904,C0103
"""
...

"""
from __future__ import division, absolute_import, print_function

import numpy as np

from km3pipe.testing import TestCase
from km3pipe.dataclasses import Hit, Track, Position, Direction_, HitSeries

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestPosition(TestCase):

    def test_position(self):
        position = Position((1., 2, 3))
        self.assertAlmostEqual(1, position.x)
        self.assertAlmostEqual(2, position.y)
        self.assertAlmostEqual(3, position.z)

    def test_attributes_can_be_changed(self):
        position = Position((4., 5, 6))
        position.x = 1.
        position.y = 2.
        position.z = 3.
        self.assertAlmostEqual(1, position.x)
        # self.assertAlmostEqual(1, position[0])
        self.assertAlmostEqual(2, position.y)
        # self.assertAlmostEqual(2, position[1])
        self.assertAlmostEqual(3, position.z)
        # self.assertAlmostEqual(3, position[2])

#   def test_position_is_ndarray_like(self):
#       pos = Position((1., 2, 3))
#       pos *= 2
#       self.assertAlmostEqual(4, pos[1])
#       self.assertAlmostEqual(3, pos.size)
#       self.assertTupleEqual((3,), pos.shape)


class TestDirection(TestCase):

    def test_direction(self):
        direction = Direction_((1, 0, 0))
        self.assertAlmostEqual(1, np.linalg.norm(direction))
        self.assertEqual(1, direction.x)
        self.assertEqual(0, direction.y)
        self.assertEqual(0, direction.z)

    def test_direction_normalises_on_init(self):
        direction = Direction_((1, 2, 3))
        self.assertAlmostEqual(0.26726124, direction.x)
        self.assertAlmostEqual(0.53452248, direction.y)
        self.assertAlmostEqual(0.80178372, direction.z)

    def test_direction_normalises_on_change_attribute(self):
        direction = Direction_((1, 2, 3))
        self.assertAlmostEqual(1, np.linalg.norm(direction))
        direction.x = 10
        self.assertAlmostEqual(1, np.linalg.norm(direction))
        direction.y = 20
        self.assertAlmostEqual(1, np.linalg.norm(direction))
        direction.z = 30
        self.assertAlmostEqual(1, np.linalg.norm(direction))

    def test_direction_zenith(self):
        direction = Direction_((0, 0, -1))
        self.assertAlmostEqual(0, direction.zenith)
        direction = Direction_((0, 0, 1))
        self.assertAlmostEqual(np.pi, direction.zenith)
        direction = Direction_((0, 1, 0))
        self.assertAlmostEqual(np.pi/2, direction.zenith)

    def test_direction_str(self):
        direction = Direction_((1, 2, 3))
        self.assertEqual("(0.2673, 0.5345, 0.8018)", str(direction))


class TestHitSeries(TestCase):

    def test_from_arrays(self):
        n = 10
        ids = np.array(range(n))
        dom_ids = np.array(range(n))
        times = np.array(range(n))
        tots = np.array(range(n))
        channel_ids = np.array(range(n))
        triggereds = np.ones(n)
        pmt_ids = np.array(range(n))

        hits = HitSeries.from_arrays(ids, dom_ids, times, tots, channel_ids,
                                     triggereds, pmt_ids)

        self.assertAlmostEqual(1, hits[1].id)
        self.assertAlmostEqual(9, hits[9].pmt_id)
        self.assertEqual(10, len(hits))


class TestHit(TestCase):

    def setUp(self):
        self.hit = Hit(1, 2, 3, 4, 5, 6, True)

    def test_attributes(self):
        hit = self.hit
        self.assertAlmostEqual(1, hit.channel_id)
        self.assertAlmostEqual(2, hit.dom_id)
        self.assertAlmostEqual(3, hit.id)
        self.assertAlmostEqual(4, hit.pmt_id)
        self.assertAlmostEqual(5, hit.time)
        self.assertAlmostEqual(6, hit.tot)
        self.assertTrue(hit.triggered)

    def test_string_representation(self):
        hit = self.hit
        representation = "Hit: channel_id(1), dom_id(2), pmt_id(4), tot(6), " \
                         "time(5), triggered(True)"
        self.assertEqual(representation, str(hit))


class TestTrack(TestCase):

    def setUp(self):
        self.track = Track(0, np.array((1, 2, 3)), 4, 5, 6, True, 8,
                      np.array((9, 10, 11)), 12, 13)

    def test_attributes(self):
        track = self.track
        self.assertAlmostEqual(1, track.dir[0])
        self.assertAlmostEqual(2, track.dir[1])
        self.assertAlmostEqual(3, track.dir[2])
        self.assertAlmostEqual(4, track.energy)
        self.assertAlmostEqual(5, track.id)
        self.assertAlmostEqual(6, track.interaction_channel)
        self.assertTrue(track.is_cc)
        self.assertAlmostEqual(8, track.length)
        self.assertAlmostEqual(9, track.pos[0])
        self.assertAlmostEqual(10, track.pos[1])
        self.assertAlmostEqual(11, track.pos[2])
        self.assertAlmostEqual(12, track.time)
        self.assertAlmostEqual(13, track.type)

    def test_string_representation(self):
        track = Track(0, np.array((1, 2, 3)), 4, 5, 6, True, 8,
                      np.array((9, 10, 11)), 12, 13)
        representation = "Track: pos([ 9 10 11]), dir([1 2 3]), t=12, " \
                         "E=4.0, type=13 (mu-)"
        self.assertEqual(representation, str(track))

    def test_mutable_dir(self):
        track = self.track

        track.dir = np.array((100, 101, 102))

        self.assertAlmostEqual(100, track.dir[0])
        self.assertAlmostEqual(101, track.dir[1])
        self.assertAlmostEqual(102, track.dir[2])

    def test_mutable_pos(self):
        track = self.track

        track.pos = np.array((100, 101, 102))

        self.assertAlmostEqual(100, track.pos[0])
        self.assertAlmostEqual(101, track.pos[1])
        self.assertAlmostEqual(102, track.pos[2])
