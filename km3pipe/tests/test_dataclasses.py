# coding=utf-8
# Filename: test_dataclasses.py
"""
...

"""
from __future__ import division, absolute_import, print_function

from km3pipe.testing import *
from km3pipe.dataclasses import Position

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
