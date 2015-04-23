# coding=utf-8                    
# Filename: test_dataclasses.py
# pylint: disable=locally-disabled,C0111
"""
...

"""
from __future__ import division, absolute_import, print_function

import numpy as np

from km3pipe.testing import *
from km3pipe.dataclasses import Position, Direction


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

    def test_direction_zenith(self):
        dir = Direction((0, 0, -1))
        self.assertAlmostEqual(0, dir.zenith)
        dir = Direction((0, 0, 1))
        self.assertAlmostEqual(np.pi, dir.zenith)
        dir = Direction((0, 1, 0))
        self.assertAlmostEqual(np.pi/2, dir.zenith)

    def test_direction_str(self):
        dir = Direction((1, 2, 3))
        self.assertEqual("(0.2673, 0.5345, 0.8018)", str(dir))


