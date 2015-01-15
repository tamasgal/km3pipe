# coding=utf-8
# Filename: test_dataclasses.py
"""
...

"""
from __future__ import division, absolute_import, print_function

import numpy as np

from km3pipe.testing import *
from km3pipe.dataclasses import Position, Direction, Hit


class TestPosition(TestCase):

    def test_position(self):
        pos = Position((1, 2, 3))
        self.assertEqual(1, pos.x)
        self.assertEqual(2, pos.y)
        self.assertEqual(3, pos.z)

    def test_attributes_can_be_changed(self):
        pos = Position()
        pos.x = 1
        pos.y = 2
        pos.z = 3
        self.assertEqual(1, pos.x)
        self.assertEqual(1, pos[0])
        self.assertEqual(2, pos.y)
        self.assertEqual(2, pos[1])
        self.assertEqual(3, pos.z)
        self.assertEqual(3, pos[2])

    def test_position_is_ndarray_like(self):
        pos = Position((1, 2, 3))
        pos *= 2
        self.assertEqual(4, pos[1])
        self.assertEqual(3, pos.size)
        self.assertTupleEqual((3,), pos.shape)


class TestDirection(TestCase):

    def test_direction(self):
        dir = Direction((1, 0, 0))
        self.assertAlmostEqual(1, np.linalg.norm(dir))
        self.assertEqual(1, dir.x)
        self.assertEqual(0, dir.y)
        self.assertEqual(0, dir.z)

    def test_direction_normalises_on_init(self):
        dir = Direction((1, 2, 3))
        self.assertAlmostEqual(0.26726124, dir.x)
        self.assertAlmostEqual(0.53452248, dir.y)
        self.assertAlmostEqual(0.80178372, dir.z)

    def test_direction_normalises_on_change_attribute(self):
        dir = Direction((1, 2, 3))
        self.assertAlmostEqual(1, np.linalg.norm(dir))
        dir.x = 10
        self.assertAlmostEqual(1, np.linalg.norm(dir))
        dir.y = 20
        self.assertAlmostEqual(1, np.linalg.norm(dir))
        dir.z = 30
        self.assertAlmostEqual(1, np.linalg.norm(dir))


class TestHit(TestCase):

    def test_hit_init(self):
        hit = Hit(1, 2, 3, True)
        self.assertEqual(1, hit.id)
        self.assertEqual(2, hit.pmt_id)
        self.assertEqual(3, hit.time)
        self.assertTrue(hit.is_noise)

    def test_hit_default_values(self):
        hit = Hit()
        self.assertIsNone(hit.id)
        self.assertIsNone(hit.pmt_id)
        self.assertIsNone(hit.time)
        self.assertIsNone(hit.is_noise)

    def test_hit_default_values_are_set_if_others_are_given(self):
        hit = Hit(is_noise=True)
        self.assertTrue(hit.is_noise)
        self.assertIsNone(hit.id)
        self.assertIsNone(hit.time)

    def test_hit_attributes_are_immutable(self):
        hit = Hit(1, True)
        with self.assertRaises(AttributeError):
            hit.time = 10
