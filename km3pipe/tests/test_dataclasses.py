# coding=utf-8
# Filename: test_dataclasses.py
# pylint: disable=C0111,R0904,C0103
"""
...

"""
from __future__ import division, absolute_import, print_function

import numpy as np

from km3pipe.testing import TestCase
from km3pipe.dataclasses import Position, Direction_, HitSeries


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
