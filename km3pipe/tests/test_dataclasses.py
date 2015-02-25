# coding=utf-8
# Filename: test_dataclasses.py
"""
...

"""
from __future__ import division, absolute_import, print_function

import operator

import numpy as np

from km3pipe.testing import *
from km3pipe.dataclasses import (Position, Direction, Track, TrackIn,
                                 TrackFit, Neutrino, Hit, RawHit)


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


class TestTrack(TestCase):
    def test_track_init(self):
        track = Track(1, 2, 3, 4, 0, 0, 1, 8, 9, 'a', 'b', 'c')
        self.assertEqual(1, track.id)
        self.assertListEqual([2, 3, 4], list(track.pos))
        self.assertListEqual([0, 0, 1], list(track.dir))
        self.assertEqual(8, track.E)
        self.assertEqual(9, track.time)
        self.assertTupleEqual(('a', 'b', 'c'), track.args)

    def test_track_repr(self):
        track = Track(1, 2, 3, 4, 0, 0, 1, 8, 9, 'a', 'b', 'c')
        repr_str = ("Track:\n id: 1\n pos: [2 3 4]\n dir: (0.0, 0.0, 1.0)\n"
                    " energy: 8 GeV\n time: 9 ns\n")
        self.assertEqual(repr_str, track.__repr__())


class TestTrackIn(TestCase):

    def test_trackin_init(self):
        track_in = TrackIn(1, 2, 3, 4, 0, 0, 1, 8, 9, 10, 11)
        self.assertEqual(1, track_in.id)
        self.assertListEqual([2, 3, 4], list(track_in.pos))
        self.assertListEqual([0, 0, 1], list(track_in.dir))
        self.assertEqual(8, track_in.E)
        self.assertEqual(9, track_in.time)
        self.assertEqual(130, track_in.particle_type) # this should be PDG!
        self.assertEqual(11, track_in.length)

    def test_track_repr(self):
        track_in = TrackIn(1, 2, 3, 4, 0, 0, 1, 8, 9, 10, 11)
        repr_str = ("Track:\n id: 1\n pos: [2 3 4]\n dir: (0.0, 0.0, 1.0)\n"
                    " energy: 8 GeV\n time: 9 ns\n type: 130 'K0L' [PDG]\n"
                    " length: 11 [m]\n")
        self.assertEqual(repr_str, track_in.__repr__())


class TestTrackFit(TestCase):

    def test_trackfit_init(self):
        track_fit = TrackFit(1, 2, 3, 4, 0, 0, 1, 8, 9, 10, 11, 12, 13, 14)
        self.assertEqual(1, track_fit.id)
        self.assertListEqual([2, 3, 4], list(track_fit.pos))
        self.assertListEqual([0, 0, 1], list(track_fit.dir))
        self.assertEqual(8, track_fit.E)
        self.assertEqual(9, track_fit.time)
        self.assertEqual(10, track_fit.speed)
        self.assertEqual(11, track_fit.ts)
        self.assertEqual(12, track_fit.te)
        self.assertEqual(13, track_fit.con1)
        self.assertEqual(14, track_fit.con2)

    def test_trackfit_repr(self):
        track_fit = TrackFit(1, 2, 3, 4, 0, 0, 1, 8, 9, 10, 11, 12, 13, 14)
        repr_str = ("Track:\n id: 1\n pos: [2 3 4]\n dir: (0.0, 0.0, 1.0)\n "
                    "energy: 8 GeV\n time: 9 ns\n speed: 10 [m/ns]\n"
                    " ts: 11 [ns]\n te: 12 [ns]\n con1: 13\n con2: 14\n")
        self.assertEqual(repr_str, track_fit.__repr__())


class TestNeutrino(TestCase):

    def test_neutrino_init(self):
        neutrino = Neutrino(1, 2, 3, 4, 0, 0, 1, 8, 9, 10, 11, 12, 13, 14)
        self.assertEqual(1, neutrino.id)
        self.assertListEqual([2, 3, 4], list(neutrino.pos))
        self.assertListEqual([0, 0, 1], list(neutrino.dir))
        self.assertEqual(8, neutrino.E)
        self.assertEqual(9, neutrino.time)
        self.assertEqual(10, neutrino.Bx)
        self.assertEqual(11, neutrino.By)
        self.assertEqual(12, neutrino.ichan)
        self.assertEqual(13, neutrino.particle_type)
        self.assertEqual(14, neutrino.channel)

    def test_neutrino_str(self):
        neutrino = Neutrino(1, 2, 3, 4, 0, 0, 1, 8, 9, 10, 11, 12, 13, 14)
        repr_str = "Neutrino: mu-, 8 GeV, NC"
        self.assertEqual(repr_str, str(neutrino))


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