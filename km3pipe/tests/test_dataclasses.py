# coding=utf-8
# Filename: test_dataclasses.py
"""
...

"""
from __future__ import division, absolute_import, print_function

import operator

import numpy as np

from km3pipe.testing import *
from km3pipe.dataclasses import Position, Direction, Hit, RawHit


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
        hit = Hit(1, 2, 3, 4, 5, 6, 7, 8)
        self.assertEqual(1, hit.id)
        self.assertEqual(2, hit.pmt_id)
        self.assertEqual(3, hit.pe)
        self.assertEqual(4, hit.time)
        self.assertEqual(5, hit.type)
        self.assertEqual(6, hit.n_photons)
        self.assertEqual(7, hit.track_in)
        self.assertEqual(8, hit.c_time)

    def test_hit_default_values(self):
        hit = Hit()
        self.assertIsNone(hit.id)
        self.assertIsNone(hit.pmt_id)
        self.assertIsNone(hit.time)

    def test_hit_default_values_are_set_if_others_are_given(self):
        hit = Hit(track_in=1)
        self.assertIsNone(hit.id)
        self.assertIsNone(hit.time)

    def test_hit_attributes_are_immutable(self):
        hit = Hit(1, True)
        with self.assertRaises(AttributeError):
            hit.time = 10




class TestRawHit(TestCase):

    def test_rawhit_init(self):
        raw_hit = RawHit(1, 2, 3, 4)
        self.assertEqual(1, raw_hit.id)
        self.assertEqual(2, raw_hit.pmt_id)
        self.assertEqual(3, raw_hit.tot)
        self.assertEqual(4, raw_hit.time)

    def test_hit_default_values(self):
        raw_hit = RawHit()
        self.assertIsNone(raw_hit.id)
        self.assertIsNone(raw_hit.pmt_id)
        self.assertIsNone(raw_hit.time)

    def test_hit_default_values_are_set_if_others_are_given(self):
        raw_hit = RawHit(pmt_id=1)
        self.assertIsNone(raw_hit.id)
        self.assertIsNone(raw_hit.time)

    def test_hit_attributes_are_immutable(self):
        raw_hit = RawHit(1, True)
        with self.assertRaises(AttributeError):
            raw_hit.time = 10

    def test_hit_addition_remains_time_id_and_pmt_id_and_adds_tot(self):
        hit1 = RawHit(id=1, time=1, pmt_id=1, tot=10)
        hit2 = RawHit(id=2, time=2, pmt_id=2, tot=20)
        merged_hit = hit1 + hit2
        self.assertEqual(1, merged_hit.id)
        self.assertEqual(1, merged_hit.time)
        self.assertEqual(1, merged_hit.pmt_id)
        self.assertEqual(30, merged_hit.tot)

    def test_hit_addition_picks_correct_time_if_second_hit_is_the_earlier_one(self):
        hit1 = RawHit(id=1, time=2, pmt_id=1, tot=10)
        hit2 = RawHit(id=2, time=1, pmt_id=2, tot=20)
        merged_hit = hit1 + hit2
        self.assertEqual(2, merged_hit.id)
        self.assertEqual(2, merged_hit.pmt_id)

    def test_hit_additions_works_with_multiple_hits(self):
        hit1 = RawHit(id=1, time=2, pmt_id=1, tot=10)
        hit2 = RawHit(id=2, time=1, pmt_id=2, tot=20)
        hit3 = RawHit(id=3, time=1, pmt_id=3, tot=30)
        merged_hit = hit1 + hit2 + hit3
        self.assertEqual(2, merged_hit.pmt_id)
        self.assertEqual(60, merged_hit.tot)
        self.assertEqual(1, merged_hit.time)
        self.assertEqual(2, merged_hit.id)

    def test_hit_addition_works_with_sum(self):
        hit1 = RawHit(id=1, time=2, pmt_id=1, tot=10)
        hit2 = RawHit(id=2, time=1, pmt_id=2, tot=20)
        hit3 = RawHit(id=3, time=1, pmt_id=3, tot=30)
        hits = [hit1, hit2, hit3]
        merged_hit = reduce(operator.add, hits)
        self.assertEqual(2, merged_hit.id)
        self.assertEqual(1, merged_hit.time)
        self.assertEqual(60, merged_hit.tot)
        self.assertEqual(2, merged_hit.pmt_id)