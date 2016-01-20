# coding=utf-8
# Filename: test_dataclasses.py
# pylint: disable=C0111,R0904,C0103
"""
...

"""
from __future__ import division, absolute_import, print_function

import numpy as np

from km3pipe.testing import TestCase
from km3pipe.dataclasses import Position, Direction


class TestPosition(TestCase):

    def test_position(self):
        position = Position((1, 2, 3))
        self.assertEqual(1, position.x)
        self.assertEqual(2, position.y)
        self.assertEqual(3, position.z)

    def test_attributes_can_be_changed(self):
        position = Position()
        position.x = 1
        position.y = 2
        position.z = 3
        self.assertEqual(1, position.x)
        self.assertEqual(1, position[0])
        self.assertEqual(2, position.y)
        self.assertEqual(2, position[1])
        self.assertEqual(3, position.z)
        self.assertEqual(3, position[2])

    def test_position_is_ndarray_like(self):
        pos = Position((1, 2, 3))
        pos *= 2
        self.assertEqual(4, pos[1])
        self.assertEqual(3, pos.size)
        self.assertTupleEqual((3,), pos.shape)


class TestDirection(TestCase):

    def test_direction(self):
        direction = Direction((1, 0, 0))
        self.assertAlmostEqual(1, np.linalg.norm(direction))
        self.assertEqual(1, direction.x)
        self.assertEqual(0, direction.y)
        self.assertEqual(0, direction.z)

    def test_direction_normalises_on_init(self):
        direction = Direction((1, 2, 3))
        self.assertAlmostEqual(0.26726124, direction.x)
        self.assertAlmostEqual(0.53452248, direction.y)
        self.assertAlmostEqual(0.80178372, direction.z)

    def test_direction_normalises_on_change_attribute(self):
        direction = Direction((1, 2, 3))
        self.assertAlmostEqual(1, np.linalg.norm(direction))
        direction.x = 10
        self.assertAlmostEqual(1, np.linalg.norm(direction))
        direction.y = 20
        self.assertAlmostEqual(1, np.linalg.norm(direction))
        direction.z = 30
        self.assertAlmostEqual(1, np.linalg.norm(direction))

    def test_direction_zenith(self):
        direction = Direction((0, 0, -1))
        self.assertAlmostEqual(0, direction.zenith)
        direction = Direction((0, 0, 1))
        self.assertAlmostEqual(np.pi, direction.zenith)
        direction = Direction((0, 1, 0))
        self.assertAlmostEqual(np.pi/2, direction.zenith)

    def test_direction_str(self):
        direction = Direction((1, 2, 3))
        self.assertEqual("(0.2673, 0.5345, 0.8018)", str(direction))
