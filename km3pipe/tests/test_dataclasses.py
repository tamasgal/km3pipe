# coding=utf-8
# Filename: test_dataclasses.py
# pylint: disable=C0111,R0904,C0103
"""
...

"""
from __future__ import division, absolute_import, print_function

from six import with_metaclass

import numpy as np
from numpy import nan
from io import BytesIO
import struct

from km3pipe.testing import TestCase
from km3pipe.testing.mocks import FakeAanetHit
from km3pipe.io.evt import EvtRawHit
from km3pipe.dataclasses import (RawHit, Hit, Track, Position,
                                 RawHitSeries, HitSeries, McHitSeries,
                                 TimesliceHitSeries,
                                 EventInfo, SummarysliceInfo, TimesliceInfo,
                                 Serialisable, TrackSeries, SummaryframeSeries,
                                 KM3Array, KM3DataFrame, BinaryStruct,
                                 BinaryComposite, DTypeAttr)

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestSerialisableABC(TestCase):

    def test_dtype_can_be_set(self):
        class TestClass(with_metaclass(Serialisable)):
            dtype = [('a', '<i4'), ('b', '>i8')]

        self.assertTupleEqual(('a', 'b'), TestClass.dtype.names)

    def test_dtype_raises_type_error_for_invalid_dtype(self):
        with self.assertRaises(TypeError):
            class TestClass(with_metaclass(Serialisable)):
                dtype = 1

    def test_arguments_are_set_correctly_as_attributes(self):
        class TestClass(with_metaclass(Serialisable)):
            dtype = [('a', '<i4'), ('b', '>i8')]

        t = TestClass(1, 2)
        self.assertEqual(1, t.a)
        self.assertEqual(2, t.b)

    def test_keyword_arguments_are_set_correctly_as_attributes(self):
        class TestClass(with_metaclass(Serialisable)):
            dtype = [('a', '<i4'), ('b', '>i8')]

        t = TestClass(b=1, a=2)
        self.assertEqual(2, t.a)
        self.assertEqual(1, t.b)

    def test_mixed_arguments_are_set_correctly_as_attributes(self):
        class TestClass(with_metaclass(Serialisable)):
            dtype = [('a', '<i4'), ('b', '>i8'), ('c', '<i4'), ('d', '<i4')]

        t = TestClass(1, 2, d=3, c=4)
        self.assertEqual(1, t.a)
        self.assertEqual(2, t.b)
        self.assertEqual(4, t.c)
        self.assertEqual(3, t.d)

    def test_setting_undefined_attribute(self):
        # TODO: discuss what should happen, currently it passes silently

        class TestClass(with_metaclass(Serialisable)):
            dtype = [('a', '<i4')]

        t = TestClass(b=3)
        self.assertEqual(3, t.b)


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


class TestTimesliceHitSeries(TestCase):

    def test_from_arrays(self):
        n = 10
        dom_ids = np.arange(n)
        times = np.arange(n)
        tots = np.arange(n)
        channel_ids = np.arange(n)

        hits = TimesliceHitSeries.from_arrays(
            channel_ids,
            dom_ids,
            times,
            tots,
            42,      # slice_id
            23,      # frame_id
        )

        self.assertAlmostEqual(1, hits[1].channel_id)
        self.assertAlmostEqual(9, hits[9].tot)
        self.assertEqual(10, len(hits))
        self.assertEqual(42, hits.slice_id)
        self.assertEqual(23, hits.frame_id)


class TestHitSeries(TestCase):
    def setUp(self):
        n = 10
        ids = np.arange(n)
        dom_ids = np.arange(n)
        dir_xs = np.arange(n)
        dir_ys = np.arange(n)
        dir_zs = np.arange(n)
        pos_xs = np.arange(n)
        pos_ys = np.arange(n)
        pos_zs = np.arange(n)
        t0s = np.arange(n)
        times = np.arange(n)
        tots = np.arange(n)
        channel_ids = np.arange(n)
        triggereds = np.ones(n)
        pmt_ids = np.arange(n)

        self.hits = HitSeries.from_arrays(
            channel_ids,
            dir_xs,
            dir_ys,
            dir_zs,
            dom_ids,
            ids,
            pos_xs,
            pos_ys,
            pos_zs,
            pmt_ids,
            t0s,
            times,
            tots,
            triggereds,
            0,      # event_id
        )

    def test_from_arrays(self):
        hits = self.hits
        self.assertAlmostEqual(1, hits[1].id)
        self.assertAlmostEqual(9, hits[9].pmt_id)
        self.assertEqual(10, len(hits))

    def test_uncalib_hits_dont_have_pmt_info(self):
        n = 10
        nans = np.full(n, np.nan, dtype='<f8')
        ids = np.arange(n)
        dom_ids = np.arange(n)
        times = np.arange(n)
        tots = np.arange(n)
        channel_ids = np.arange(n)
        triggereds = np.ones(n)
        pmt_ids = np.arange(n)

        hits = HitSeries.from_arrays(
            channel_ids,
            nans, nans, nans,
            dom_ids,
            ids,
            pmt_ids,
            nans, nans, nans, 0,
            times,
            tots,
            triggereds,
            0,      # event_id
        )

        self.assertAlmostEqual(1, hits[1].id)
        self.assertAlmostEqual(9, hits[9].pmt_id)
        self.assertEqual(10, len(hits))

    def test_from_aanet(self):
        n_params = 16
        n_hits = 10
        hits = [FakeAanetHit(*p) for p in
                np.arange(n_hits * n_params).reshape(n_hits, n_params)]
        hit_series = HitSeries.from_aanet(hits, 0)

        self.assertAlmostEqual(6, hit_series.pmt_id[0])
        self.assertAlmostEqual(6, hit_series[0].pmt_id)
        self.assertAlmostEqual(16, hit_series[1].channel_id)
        self.assertAlmostEqual(53, hit_series[3].id)
        self.assertAlmostEqual(92, hit_series[5].tot)
        self.assertTrue(hit_series[9].triggered)
        self.assertAlmostEqual(116, hit_series[7].dom_id)
        self.assertAlmostEqual(155, hit_series[9].time)
        self.assertEqual(n_hits, len(hit_series))

    def test_attributes_via_from_aanet(self):
        n_params = 16
        n_hits = 10
        hits = [FakeAanetHit(*p) for p in
                np.arange(n_hits * n_params).reshape(n_hits, n_params)]
        hit_series = HitSeries.from_aanet(hits, 0)

        self.assertTupleEqual((5, 21, 37, 53, 69, 85, 101, 117, 133, 149),
                              tuple(hit_series.id))
        self.assertTupleEqual(
            (4, 20, 36, 52, 68, 84, 100, 116, 132, 148),
            tuple(hit_series.dom_id))
        self.assertTupleEqual(
            (6, 22, 38, 54, 70, 86, 102, 118, 134, 150),
            tuple(hit_series.pmt_id))
        self.assertTupleEqual(
            (0, 16, 32, 48, 64, 80, 96, 112, 128, 144),
            tuple(hit_series.channel_id))
        self.assertTupleEqual(
            (11, 27, 43, 59, 75, 91, 107, 123, 139, 155),
            tuple(hit_series.time))
        self.assertTupleEqual(
            # triggered is an unsigned short integer
            (13, 29, 45, 61, 77, 93, 109, 125, 141, 157),
            tuple(hit_series.triggered))
        self.assertTupleEqual(
            (12, 28, 44, 60, 76, 92, 108, 124, 140, 156),
            tuple(hit_series.tot))

    def test_from_evt(self):
        n_params = 4
        n_hits = 10
        hits = [EvtRawHit(*p) for p in
                np.arange(n_hits * n_params).reshape(n_hits, n_params)]
        print(len(hits))
        print(len(hits[0]))
        hit_series = HitSeries.from_evt(hits, 0)

        self.assertAlmostEqual(1, hit_series[0].pmt_id)
        self.assertAlmostEqual(0, hit_series[1].channel_id)  # always 0 for MC
        self.assertAlmostEqual(12, hit_series[3].id)
        self.assertAlmostEqual(22, hit_series[5].tot)
        self.assertFalse(hit_series[2].triggered)  # always False for MC
        self.assertAlmostEqual(0, hit_series[7].dom_id)  # always 0 for MC
        self.assertAlmostEqual(39, hit_series[9].time)
        self.assertEqual(n_hits, len(hit_series))

    def test_attributes_via_from_evt(self):
        n_params = 4
        n_hits = 10
        hits = [EvtRawHit(*p) for p in
                np.arange(n_hits * n_params).reshape(n_hits, n_params)]
        hit_series = HitSeries.from_evt(hits, 0)

        self.assertTupleEqual((0, 4, 8, 12, 16, 20, 24, 28, 32, 36),
                              tuple(hit_series.id))
        self.assertTupleEqual((0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                              tuple(hit_series.dom_id))
        self.assertTupleEqual((1, 5, 9, 13, 17, 21, 25, 29, 33, 37),
                              tuple(hit_series.pmt_id))
        self.assertTupleEqual((0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                              tuple(hit_series.channel_id))
        self.assertTupleEqual((3, 7, 11, 15, 19, 23, 27, 31, 35, 39),
                              tuple(hit_series.time))
        self.assertTupleEqual((False, False, False, False, False, False, False,
                               False, False, False),
                              tuple(hit_series.triggered))
        self.assertTupleEqual((2, 6, 10, 14, 18, 22, 26, 30, 34, 38),
                              tuple(hit_series.tot))

    def test_first_hits_unique(self):
        n_fh = len(self.hits.first_hits)
        n_unique = len(np.unique(self.hits.dom_id))
        self.assertEqual(n_fh, n_unique)


class TestRawHitSeries(TestCase):
    def setUp(self):
        n = 10
        channel_ids = np.arange(n)
        dom_ids = np.arange(n)
        times = np.arange(n)
        tots = np.arange(n)
        triggereds = np.ones(n)

        self.hits = RawHitSeries.from_arrays(
            channel_ids,
            dom_ids,
            times,
            tots,
            triggereds,
            0,      # event_id
        )

    def test_from_arrays(self):
        hits = self.hits
        self.assertAlmostEqual(1, hits[1].channel_id)
        self.assertAlmostEqual(9, hits[9].dom_id)
        self.assertAlmostEqual(9, hits[9].time)
        self.assertAlmostEqual(9, hits[9].tot)
        self.assertAlmostEqual(1, hits[9].triggered)
        self.assertEqual(10, len(hits))

    def test_single_element_attr_access(self):
        hits = self.hits
        a_hit = hits[2]
        self.assertAlmostEqual(2, a_hit.channel_id)
        self.assertAlmostEqual(2, a_hit.dom_id)
        self.assertAlmostEqual(2, a_hit.time)
        self.assertAlmostEqual(2, a_hit.tot)
        self.assertAlmostEqual(1, a_hit.triggered)

    def test_slicing(self):
        hits = self.hits
        shits = self.hits[4:6]
        self.assertTrue(isinstance(shits, RawHitSeries))
        self.assertEqual(2, len(shits))
        self.assertAlmostEqual(4, shits.time[0])

    def test_slicing_then_single_element_access(self):
        hits = self.hits
        shits = self.hits[4:6]
        a_hit = shits[1]
        self.assertAlmostEqual(5, a_hit.time)

    def test_appending_fields(self):
        hits = self.hits
        hits.append_fields('new', [1, 2, 3, 4])
        self.assertEqual(1, hits.new[0])

    def test_appending_fields_survives_slicing(self):
        hits = self.hits
        hits.append_fields('new', [1, 2, 3, 4])
        shits = hits[2:4]
        self.assertTrue(isinstance(shits, RawHitSeries))
        self.assertEqual(3, shits.new[0])

    def test_from_aanet(self):
        n_params = 16
        n_hits = 10
        hits = [FakeAanetHit(*p) for p in
                np.arange(n_hits * n_params).reshape(n_hits, n_params)]
        hit_series = RawHitSeries.from_aanet(hits, 0)

        self.assertAlmostEqual(16, hit_series[1].channel_id)
        self.assertAlmostEqual(92, hit_series[5].tot)
        self.assertTrue(hit_series[9].triggered)
        self.assertAlmostEqual(116, hit_series[7].dom_id)
        self.assertAlmostEqual(155, hit_series[9].time)
        self.assertEqual(n_hits, len(hit_series))

    def test_attributes_via_from_aanet(self):
        n_params = 16
        n_hits = 10
        hits = [FakeAanetHit(*p) for p in
                np.arange(n_hits * n_params).reshape(n_hits, n_params)]
        hit_series = RawHitSeries.from_aanet(hits, 0)

        self.assertTupleEqual(
            (4, 20, 36, 52, 68, 84, 100, 116, 132, 148),
            tuple(hit_series.dom_id))
        self.assertTupleEqual(
            (0, 16, 32, 48, 64, 80, 96, 112, 128, 144),
            tuple(hit_series.channel_id))
        self.assertTupleEqual(
            (11, 27, 43, 59, 75, 91, 107, 123, 139, 155),
            tuple(hit_series.time))
        self.assertTupleEqual(
            # triggered is an unsigned short integer
            (13, 29, 45, 61, 77, 93, 109, 125, 141, 157),
            tuple(hit_series.triggered))
        self.assertTupleEqual(
            (12, 28, 44, 60, 76, 92, 108, 124, 140, 156),
            tuple(hit_series.tot))


class TestMcHitSeries(TestCase):
    def setUp(self):
        n = 10
        a = np.arange(n)
        origin = np.arange(n)
        pmt_id = np.arange(n)
        time = np.ones(n)

        self.hits = McHitSeries.from_arrays(
            a,
            origin,
            pmt_id,
            time,
            0,      # event_id
        )

    def test_from_arrays(self):
        hits = self.hits
        self.assertAlmostEqual(9, hits[9].pmt_id)
        self.assertAlmostEqual(1.0, hits[9].time)
        self.assertAlmostEqual(9, hits[9].a)
        self.assertAlmostEqual(9, hits[9].origin)
        self.assertEqual(10, len(hits))

    def test_from_aanet(self):
        n_params = 16
        n_hits = 10
        hits = [FakeAanetHit(*p) for p in
                np.arange(n_hits * n_params).reshape(n_hits, n_params)]
        hit_series = McHitSeries.from_aanet(hits, 0)

        self.assertAlmostEqual(22, hit_series[1].pmt_id)
        self.assertAlmostEqual(94, hit_series[5].a)
        self.assertTrue(hit_series[9].origin)
        self.assertAlmostEqual(155, hit_series[9].time)
        self.assertEqual(n_hits, len(hit_series))

    def test_attributes_via_from_aanet(self):
        n_params = 16
        n_hits = 10
        hits = [FakeAanetHit(*p) for p in
                np.arange(n_hits * n_params).reshape(n_hits, n_params)]
        hit_series = McHitSeries.from_aanet(hits, 0)

        self.assertTupleEqual(
            (6, 22, 38, 54, 70, 86, 102, 118, 134, 150),
            tuple(hit_series.pmt_id))
        self.assertTupleEqual(
            (11.0, 27.0, 43.0, 59.0, 75.0, 91.0, 107.0, 123.0, 139.0, 155.0),
            tuple(hit_series.time))
        self.assertTupleEqual(
            # triggered is an unsigned short integer
            (15, 31, 47, 63, 79, 95, 111, 127, 143, 159),
            tuple(hit_series.origin))
        self.assertTupleEqual(
            (14.0, 30.0, 46.0, 62.0, 78.0, 94.0, 110.0, 126.0, 142.0, 158.0),
            tuple(hit_series.a))


class TestHit(TestCase):

    def setUp(self):
        self.hit = Hit(1, nan, nan, nan, 2, 3, 4,
                       nan, nan, nan, 0, 5, 6, 1)

    def test_attributes(self):
        hit = self.hit
        self.assertAlmostEqual(1, hit.channel_id)
        self.assertTrue(np.isnan(hit.dir_x))
        self.assertTrue(np.isnan(hit.dir_y))
        self.assertTrue(np.isnan(hit.dir_z))
        self.assertAlmostEqual(2, hit.dom_id)
        self.assertAlmostEqual(3, hit.id)
        self.assertAlmostEqual(4, hit.pmt_id)
        self.assertTrue(np.isnan(hit.pos_x))
        self.assertTrue(np.isnan(hit.pos_y))
        self.assertTrue(np.isnan(hit.pos_z))
        self.assertAlmostEqual(0, hit.t0)
        self.assertAlmostEqual(5, hit.time)
        self.assertAlmostEqual(6, hit.tot)
        self.assertAlmostEqual(1, hit.triggered)

    def test_string_representation(self):
        hit = self.hit
        representation = "Hit: channel_id(1), dom_id(2), pmt_id(4), tot(6), " \
                         "time(5), triggered(1)"
        self.assertEqual(representation, str(hit))


class TestRawHit(TestCase):

    def setUp(self):
        self.hit = RawHit(1, 2, 3, 4, True)

    def test_attributes(self):
        hit = self.hit
        self.assertAlmostEqual(1, hit.channel_id)
        self.assertAlmostEqual(2, hit.dom_id)
        self.assertAlmostEqual(3, hit.time)
        self.assertAlmostEqual(4, hit.tot)
        self.assertAlmostEqual(True, hit.triggered)

    def test_string_representation(self):
        hit = self.hit
        representation = "RawHit: channel_id(1), dom_id(2), tot(4), " \
                         "time(3), triggered(1)"
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


class TestTrackSeries(TestCase):
    def test_from_arrays(self):
        n = 10
        bjorkenys = np.array(range(n))
        dir_xs = np.array(range(n))
        dir_ys = np.array(range(n))
        dir_zs = np.array(range(n))
        energys = np.array(range(n))
        ids = np.array(range(n))
        interaction_channels = np.array(range(n))
        is_ccs = np.array([True] * 10)
        lengths = np.array(range(n))
        pos_xs = np.array(range(n))
        pos_ys = np.array(range(n))
        pos_zs = np.array(range(n))
        times = np.array(range(n))
        types = np.array(range(n))

        tracks = TrackSeries.from_arrays(
            bjorkenys,
            dir_xs,
            dir_ys,
            dir_zs,
            energys,
            ids,
            interaction_channels,
            is_ccs,
            lengths,
            pos_xs,
            pos_ys,
            pos_zs,
            times,
            types,
            event_id=0,
        )

        self.assertAlmostEqual(1, tracks[1].id)
        self.assertAlmostEqual(9, tracks[9].interaction_channel)
        self.assertEqual(10, len(tracks))

    def test_array(self):
        ts = TrackSeries.from_table([{
            'bjorkeny': 0.0,
            'dir_x': 1.0,
            'dir_y': 2.0,
            'dir_z': 3.0,
            'energy': 4.0,
            'id': 5,
            'interaction_channel': 7,
            'is_cc': True,
            'length': 9.0,
            'pos_x': 10.0,
            'pos_y': 11.0,
            'pos_z': 12.0,
            'time': 13,
            'type': 14,
        }], event_id=0)
        exp = [(0.0, 1.0, 2.0, 3, 4.0, 0, 6, 7, True, 9.0, 10.0, 12.0, 12.0,
                13, 14), ]
        exp = np.array(exp, dtype=ts.dtype)
        self.assertEqual(1, len(ts))
        # self.assertAlmostEqual(exp, ts.serialise())


class TestSummaryframeSeries(TestCase):

    def test_from_arrays(self):
        n = 10
        dom_ids = np.arange(n)
        max_sequence_numbers = np.arange(n)
        n_received_packets = np.arange(n)
        frames = SummaryframeSeries.from_arrays(
            dom_ids,
            max_sequence_numbers,
            n_received_packets,
            23,      # slice_id
        )

        self.assertAlmostEqual(1, frames[1][0])  # dom_id
        self.assertAlmostEqual(9, frames[9][2])  # n_received_packets
        self.assertTupleEqual(tuple(range(n)),
                              tuple(frames.n_received_packets))
        self.assertEqual(10, len(frames))


class TestTimesliceInfo(TestCase):
    def test_timeslice_info(self):
        s = TimesliceInfo(frame_index=0, slice_id=1, timestamp=3,
                          nanoseconds=4, n_frames=5)
        self.assertAlmostEqual(0, s.frame_index)
        self.assertAlmostEqual(1, s.slice_id)
        self.assertAlmostEqual(3, s.timestamp)
        self.assertAlmostEqual(4, s.nanoseconds)
        self.assertAlmostEqual(5, s.n_frames)


class TestSummarysliceInfo(TestCase):
    def test_timeslice_info(self):
        s = SummarysliceInfo(frame_index=0, slice_id=1, timestamp=3,
                             nanoseconds=4, n_frames=5)
        self.assertAlmostEqual(0, s.frame_index)
        self.assertAlmostEqual(1, s.slice_id)
        self.assertAlmostEqual(3, s.timestamp)
        self.assertAlmostEqual(4, s.nanoseconds)
        self.assertAlmostEqual(5, s.n_frames)



class TestEventInfo(TestCase):
    def test_event_info(self):
        ran = np.array(tuple(range(17)), dtype=EventInfo.dtype)
        e = EventInfo(ran)
        print(e.trigger_counter)
        print(e)
        self.assertAlmostEqual(0, e.det_id)
        self.assertAlmostEqual(1, e.frame_index)
        self.assertAlmostEqual(2, e.livetime_sec)
        self.assertAlmostEqual(3, e.mc_id)
        self.assertAlmostEqual(4, e.mc_t)
        self.assertAlmostEqual(5, e.n_events_gen)
        self.assertAlmostEqual(6, e.n_files_gen)
        self.assertAlmostEqual(7, e.overlays)
        self.assertAlmostEqual(8, e.trigger_counter)
        self.assertAlmostEqual(9, e.trigger_mask)
        self.assertAlmostEqual(10, e.utc_nanoseconds)
        self.assertAlmostEqual(11, e.utc_seconds)
        self.assertAlmostEqual(12, e.weight_w1)
        self.assertAlmostEqual(13, e.weight_w2)
        self.assertAlmostEqual(14, e.weight_w3)
        self.assertAlmostEqual(15, e.run_id)
        self.assertAlmostEqual(16, e.event_id)

    def test_from_table(self):
        e = EventInfo.from_row({
            'det_id': 0,
            'frame_index': 2,
            'mc_id': 3,
            'mc_t': 4,
            'overlays': 5,
            'trigger_counter': 6,
            'trigger_mask': 7,
            'utc_nanoseconds': 8,
            'utc_seconds': 9,
            'weight_w1': 10,
            'weight_w2': 11,
            'weight_w3': 12,
            'livetime_sec': 13,
            'n_events_gen': 14,
            'n_files_gen': 15,
            'run_id': 16,
            'event_id': 1,
        })

        self.assertAlmostEqual(0, e.det_id)
        self.assertAlmostEqual(2, e.frame_index)
        self.assertAlmostEqual(3, e.mc_id)
        self.assertAlmostEqual(4, e.mc_t)
        self.assertAlmostEqual(5, e.overlays)
        self.assertAlmostEqual(16, e.run_id)
        self.assertAlmostEqual(6, e.trigger_counter)
        self.assertAlmostEqual(7, e.trigger_mask)
        self.assertAlmostEqual(8, e.utc_nanoseconds)
        self.assertAlmostEqual(9, e.utc_seconds)
        self.assertAlmostEqual(10, e.weight_w1)
        self.assertAlmostEqual(11, e.weight_w2)
        self.assertAlmostEqual(12, e.weight_w3)
        self.assertAlmostEqual(1, e.event_id)
        self.assertAlmostEqual(13, e.livetime_sec)
        self.assertAlmostEqual(14, e.n_events_gen)
        self.assertAlmostEqual(15, e.n_files_gen)

    def test_array(self):
        e = EventInfo.from_row({
            'det_id': 0,
            'frame_index': 2,
            'livetime_sec': 13,
            'mc_id': 3,
            'mc_t': 4.0,
            'n_events_gen': 14,
            'n_files_gen': 15,
            'overlays': 5,
            'trigger_counter': 6,
            'trigger_mask': 7,
            'utc_nanoseconds': 8,
            'utc_seconds': 9,
            'weight_w1': 10.0,
            'weight_w2': 11.0,
            'weight_w3': 12.0,
            'run_id': 16,
            'event_id': 1,
        })
        exp = (0, 2, 13, 3, 4.0, 14, 15, 5, 6, 7, 8, 9, 10.0, 11.0, 12.0, 16, 1,)
        self.assertAlmostEqual(e.serialise(), np.array(exp, e.dtype))

    def test_missing_run_id(self):
        dt = EventInfo.dtype
        fields = dict(dt.descr)
        del fields['run_id']
        sparse_dt = np.dtype([(k, v) for k, v in fields.items()])
        e = np.ones(1, dtype=sparse_dt)
        print(e)
        print(e.dtype)
        print('run_id' in e.dtype.fields)
        ei = EventInfo(e)
        self.assertAlmostEqual(1, ei.det_id)
        self.assertAlmostEqual(1, ei.frame_index)
        self.assertAlmostEqual(1, ei.mc_id)
        self.assertAlmostEqual(1, ei.mc_t)
        self.assertAlmostEqual(1, ei.overlays)
        self.assertAlmostEqual(1, ei.trigger_counter)
        self.assertAlmostEqual(1, ei.trigger_mask)
        self.assertAlmostEqual(1, ei.utc_nanoseconds)
        self.assertAlmostEqual(1, ei.utc_seconds)
        self.assertAlmostEqual(1, ei.weight_w1)
        self.assertAlmostEqual(1, ei.weight_w2)
        self.assertAlmostEqual(1, ei.weight_w3)
        self.assertAlmostEqual(1, ei.event_id)
        self.assertAlmostEqual(1, ei.livetime_sec)
        self.assertAlmostEqual(1, ei.n_events_gen)
        self.assertAlmostEqual(1, ei.n_files_gen)
        self.assertAlmostEqual(0, ei.run_id)


class TestKM3Array(TestCase):
    def test_km3array(self):
        dt = np.dtype(sorted([('x', int), ('y', float),
                              ('did_converge', bool)]))
        dat = {'x': 4, 'y': 2.0, 'did_converge': True}
        rec = KM3Array.from_dict(dat, dtype=dt)
        print(rec)
        print(rec.dtype)
        self.assertAlmostEqual(rec.dtype, dt)
        self.assertTrue(rec['did_converge'])
        self.assertAlmostEqual(rec['x'], 4)

    def test_km3array_serialise(self):
        dt = np.dtype(sorted([('x', int), ('y', float),
                              ('did_converge', bool)]))
        dat = {'x': 4, 'y': 2.0, 'did_converge': True}
        rec = KM3Array.from_dict(dat, dtype=dt).serialise()
        print(rec)
        print(rec.dtype)
        self.assertTrue(rec[0][0])
        self.assertAlmostEqual(rec[0][1], 4)
        self.assertAlmostEqual(rec[0][2], 2.0)


class TestKM3DataFrame(TestCase):
    def test_h5loc_is_preserved_along_trafo(self):
        arr = np.random.normal(size=24).reshape(-1, 3)
        df = KM3DataFrame(arr)
        self.assertTrue(hasattr(df, 'h5loc'))
        self.assertEqual('/', df.h5loc)
        df.h5loc = '/reco'
        self.assertEqual('/reco', df.h5loc)
        self.assertEqual('/reco', df[[0, 1]].h5loc)


class TestBinaryStruct(TestCase):
    def test_init(self):
        stream = BytesIO(b'')
        b = BinaryStruct(stream)        # noqa

    def test_parsing(self):
        structure = '<2i3fc'
        values = (1, 2, 3.4, 5.6, 7.8, b'a')
        fields = 'abcdef'

        data = BytesIO(struct.pack(structure, *values))

        class Dummy(BinaryStruct):
            _structure = structure
            _fields = fields

        dummy = Dummy(data)
        self.assertEqual(1, dummy.a)
        self.assertEqual(2, dummy.b)
        self.assertAlmostEqual(3.4, dummy.c, 5)
        self.assertAlmostEqual(5.6, dummy.d, 5)
        self.assertAlmostEqual(7.8, dummy.e, 5)
        self.assertEqual(b'a', dummy.f)


class TestBinaryComposite(TestCase):
    def test_init(self):
        stream = BytesIO(b'')
        b = BinaryComposite(stream)     # noqa


class TestDTypeAttr(TestCase):
    def test_subclassing(self):
        class Foo(DTypeAttr):
            pass

    def test_access_attribute(self):
        class Foo(DTypeAttr):
            def __init__(self):
                self.dtype = lambda x: x  # quick hack to add subattr
                self.dtype.names = ['bar']
                self._arr = {"bar": 23}

        foo = Foo()
        self.assertEqual(23, foo.bar)

    def test_access_missing_attribute(self):
        class Foo(DTypeAttr):
            def __init__(self):
                self.dtype = lambda x: x  # quick hack to add subattr
                self.dtype.names = ['bar']
                self._arr = {"bar": 23}

        foo = Foo()
        with self.assertRaises(AttributeError):
            foo.baz

    def test_subclassing_a_class_without_dtype(self):
        class Foo(DTypeAttr):
            def __init__(self):
                pass

        with self.assertRaises(AttributeError):
            foo = Foo()
            foo.bar

    def test_adding_new_attribute(self):

        data = np.array([(1.0, 2), (3.0, 4)], dtype=[('x', float), ('y', int)])

        class Foo(DTypeAttr):
            dtype = data.dtype
            def __init__(self):
                self._arr = data

        foo = Foo()

        self.assertAlmostEqual(1.0, foo.x[0])
        self.assertAlmostEqual(3.0, foo.x[1])
        self.assertEqual(2, foo.y[0])
        self.assertEqual(4, foo.y[1])
    
        foo.append_fields('new', [5, 6])
        self.assertAlmostEqual(1.0, foo.x[0])
        self.assertAlmostEqual(3.0, foo.x[1])
        self.assertEqual(2, foo.y[0])
        self.assertEqual(4, foo.y[1])
        self.assertEqual(5, foo.new[0])
        self.assertEqual(6, foo.new[1])

    def test_adding_new_attribute_keeps_dtype_when_slicing(self):
        data = np.array([(1.0, 2), (3.0, 4)], dtype=[('x', float), ('y', int)])

        class Foo(DTypeAttr):
            dtype = data.dtype
            def __init__(self, data):
                self._arr = data

        foo = Foo(data)
        foo.append_fields('new', [5, 6])
        self.assertAlmostEqual(1.0, foo.x[0])
        self.assertAlmostEqual(3.0, foo.x[1])
        self.assertAlmostEqual(5, foo.new[0])
        sliced_foo = foo[1:]
        self.assertAlmostEqual(6, sliced_foo.new[0])
