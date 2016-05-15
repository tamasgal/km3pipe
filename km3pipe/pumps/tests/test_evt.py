# coding=utf-8
# Filename: test_evt.py
# pylint: disable=locally-disabled,C0111,R0904,C0301,C0103,W0212
from __future__ import division, absolute_import, print_function

import operator
from functools import reduce

from km3pipe.testing import TestCase, StringIO
from km3pipe.pumps import EvtPump
from km3pipe.pumps.evt import Track, TrackIn, Neutrino, EvtHit, EvtRawHit, TrackFit

__author__ = 'tamasgal'


class TestEvtParser(TestCase):

    def setUp(self):
        self.valid_evt_header = "\n".join((
            "start_run: 1",
            "cut_nu: 0.100E+03 0.100E+09-0.100E+01 0.100E+01",
            "spectrum: -1.40",
            "end_event:",
            "start_event: 12 1",
            "track_in: 1 -389.951 213.427 516 -0.204562 -0.60399 -0.770293" +
            " 9.092 0 5 40.998",
            "hit: 1 44675 1 1170.59 5 2 1 1170.59",
            "end_event:",
            "start_event: 13 1",
            "track_in:  1 272.695 -105.613 516 -0.425451 -0.370522 -0.825654" +
            " 2431.47 0 5 -1380",
            "track_in:  2 272.348 -106.292 516 -0.425451 -0.370522 -0.825654" +
            " 24670.7 1.33 5 -1484",
            "track_in:  3 279.47 -134.999 516 -0.425451 -0.370522 -0.825654" +
            " 164.586 26.7 5 601.939",
            "hit: 1 20140 1 1140.06 5 1 1 1140.06",
            "hit: 2 20159 1 1177.14 5 1 1 1177.14",
            "hit: 3 20164 1 1178.19 5 1 1 1178.19",
            "hit: 4 20170 1 1177.23 5 1 1 1177.23",
            "hit: 5 20171 2 1177.25 5 1 2 1177.25",
            "end_event:",
            "start_event: 14 1",
            "track_in:  1 40.256 -639.888 516 0.185998 0.476123 -0.859483" +
            " 10016.1 0 5 -1668",
            "hit: 1 33788 1 2202.81 5 1 1 2202.81",
            "hit: 2 33801 1 2248.95 5 1 1 2248.95",
            "hit: 3 33814 1 2249.2 5 1 1 2249.2",
            "end_event:"
        ))
        self.no_evt_header = "\n".join((
            "start_event: 12 1",
            "track_in: 1 -389.951 213.427 516 -0.204562 -0.60399 -0.770293" +
            " 9.092 0 5 40.998",
            "hit: 1 44675 1 1170.59 5 2 1 1170.59",
            "end_event:",
            "start_event: 13 1",
            "track_in:  1 272.695 -105.613 516 -0.425451 -0.370522 -0.825654" +
            " 2431.47 0 5 -1380",
            "track_in:  2 272.348 -106.292 516 -0.425451 -0.370522 -0.825654" +
            " 24670.7 1.33 5 -1484",
            "track_in:  3 279.47 -134.999 516 -0.425451 -0.370522 -0.825654" +
            " 164.586 26.7 5 601.939",
            "hit: 1 20140 1 1140.06 5 1 1 1140.06",
            "hit: 2 20159 1 1177.14 5 1 1 1177.14",
            "hit: 3 20164 1 1178.19 5 1 1 1178.19",
            "hit: 4 20170 1 1177.23 5 1 1 1177.23",
            "hit: 5 20171 2 1177.25 5 1 2 1177.25",
            "end_event:",
            "start_event: 14 1",
            "track_in:  1 40.256 -639.888 516 0.185998 0.476123 -0.859483" +
            " 10016.1 0 5 -1668",
            "hit: 1 33788 1 2202.81 5 1 1 2202.81",
            "hit: 2 33801 1 2248.95 5 1 1 2248.95",
            "hit: 3 33814 1 2249.2 5 1 1 2249.2",
            "end_event:"
        ))
        self.corrupt_evt_header = "foo"
        self.corrupt_line = "\n".join((
            "start_event: 1 1",
            "corrupt line",
            "end_event:"
            ))

        self.pump = EvtPump()
        self.pump.blob_file = StringIO(self.valid_evt_header)

    def tearDown(self):
        self.pump.blob_file.close()

    def test_parse_header(self):
        raw_header = self.pump.extract_header()
        self.assertEqual(['1'], raw_header['start_run'])
        self.assertAlmostEqual(-1.4, float(raw_header['spectrum'][0]))
        self.assertAlmostEqual(1, float(raw_header['cut_nu'][2]))

#    def test_incomplete_header_raises_value_error(self):
#        temp_file = StringIO(self.corrupt_evt_header)
#        pump = EvtPump()
#        pump.blob_file = temp_file
#        with self.assertRaises(ValueError):
#            pump.extract_header()
#        temp_file.close()

    def test_record_offset_saves_correct_offset(self):
        self.pump.blob_file = StringIO('a'*42)
        offsets = [1, 4, 9, 12, 23]
        for offset in offsets:
            self.pump.blob_file.seek(0, 0)
            self.pump.blob_file.seek(offset, 0)
            self.pump._record_offset()
        self.assertListEqual(offsets, self.pump.event_offsets)

    def test_event_offset_is_at_first_event_after_parsing_header(self):
        self.pump.extract_header()
        self.assertEqual(88, self.pump.event_offsets[0])

    def test_rebuild_offsets(self):
        self.pump.extract_header()
        self.pump._cache_offsets()
        self.assertListEqual([88, 233, 700], self.pump.event_offsets)

    def test_rebuild_offsets_without_header(self):
        self.pump.blob_file = StringIO(self.no_evt_header)
        self.pump.extract_header()
        self.pump._cache_offsets()
        self.assertListEqual([0, 145, 612], self.pump.event_offsets)

    def test_cache_enabled_triggers_rebuild_offsets(self):
        self.pump.cache_enabled = True
        self.pump.prepare_blobs()
        self.assertEqual(3, len(self.pump.event_offsets))

    def test_cache_disabled_doesnt_trigger_cache_offsets(self):
        self.pump.cache_enabled = False
        self.pump.prepare_blobs()
        self.assertEqual(1, len(self.pump.event_offsets))

    def test_get_blob_triggers_cache_offsets_if_cache_disabled(self):
        "...and asking for not indexed event"
        self.pump.cache_enabled = False
        self.pump.prepare_blobs()
        self.assertEqual(1, len(self.pump.event_offsets))
        blob = self.pump.get_blob(2)
        self.assertListEqual(['14', '1'], blob['start_event'])
        self.assertEqual(3, len(self.pump.event_offsets))

    def test_get_blob_raises_index_error_for_wrong_index(self):
        self.pump.prepare_blobs()
        with self.assertRaises(IndexError):
            self.pump.get_blob(23)

    def test_get_blob_returns_correct_event_information(self):
        self.pump.prepare_blobs()
        blob = self.pump.get_blob(0)
        self.assertTrue('raw_header' in blob)
        self.assertEqual(['1'], blob['raw_header']['start_run'])
        self.assertListEqual(['12', '1'], blob['start_event'])
        self.assertListEqual([[1.0, 44675.0, 1.0, 1170.59,
                               5.0, 2.0, 1.0, 1170.59]],
                             blob['hit'])

    def test_get_blob_returns_correct_events(self):
        self.pump.prepare_blobs()
        blob = self.pump.get_blob(0)
        self.assertListEqual(['12', '1'], blob['start_event'])
        blob = self.pump.get_blob(2)
        self.assertListEqual(['14', '1'], blob['start_event'])
        blob = self.pump.get_blob(1)
        self.assertListEqual(['13', '1'], blob['start_event'])

    def test_process_returns_correct_blobs(self):
        self.pump.prepare_blobs()
        blob = self.pump.process()
        self.assertListEqual(['12', '1'], blob['start_event'])
        blob = self.pump.process()
        self.assertListEqual(['13', '1'], blob['start_event'])
        blob = self.pump.process()
        self.assertListEqual(['14', '1'], blob['start_event'])

    def test_process_raises_stop_iteration_if_eof_reached(self):
        self.pump.prepare_blobs()
        self.pump.process()
        self.pump.process()
        self.pump.process()
        with self.assertRaises(StopIteration):
            self.pump.process()

    def test_pump_acts_as_iterator(self):
        self.pump.prepare_blobs()
        event_numbers = []
        for blob in self.pump:
            event_numbers.append(blob['start_event'][0])
        self.assertListEqual(['12', '13', '14'], event_numbers)

    def test_pump_has_len(self):
        self.pump.prepare_blobs()
        self.assertEqual(3, len(self.pump))

    def test_pump_get_item_returns_first_for_index_zero(self):
        self.pump.prepare_blobs()
        first_blob = self.pump[0]
        self.assertEqual('12', first_blob['start_event'][0])

    def test_pump_get_item_returns_correct_blob_for_index(self):
        self.pump.prepare_blobs()
        blob = self.pump[1]
        self.assertEqual('13', blob['start_event'][0])

    def test_pump_slice_generator(self):
        self.pump.prepare_blobs()
        blobs = self.pump[:]
        blobs = list(self.pump[1:3])
        self.assertEqual(2, len(blobs))
        self.assertEqual(['13', '1'], blobs[0]['start_event'])

    def test_create_blob_entry_for_line_ignores_corrupt_line(self):
        self.pump.blob_file = StringIO(self.corrupt_line)
        self.pump.extract_header()
        self.pump.prepare_blobs()
        self.pump.get_blob(0)


class TestTrack(TestCase):
    def setUp(self):
        self.track = Track((1., 2., 3., 4., 0., 0., 1., 8., 9., 'a', 'b', 'c'),
                           zed_correction=0.)

    def test_track_init(self):
        track = self.track
        self.assertAlmostEqual(1, track.id)
        self.assertAlmostEqual(2, track.pos.x)
        self.assertAlmostEqual(3, track.pos.y)
        self.assertAlmostEqual(4, track.pos.z)
        self.assertAlmostEqual(0, track.dir.x)
        self.assertAlmostEqual(0, track.dir.y)
        self.assertAlmostEqual(1, track.dir.z)
        self.assertAlmostEqual(8, track.E)
        self.assertAlmostEqual(9, track.time)
        self.assertTupleEqual(('a', 'b', 'c'), track.args)


class TestTrackIn(TestCase):

    def setUp(self):
        self.track_in = TrackIn((1, 2., 3., 4., 0., 0., 1., 8, 9, 10, 11),
                                zed_correction=0)

    def test_trackin_init(self):
        track_in = self.track_in
        self.assertEqual(1, track_in.id)
        self.assertAlmostEqual(2, track_in.pos.x)
        self.assertAlmostEqual(3, track_in.pos.y)
        self.assertAlmostEqual(4, track_in.pos.z)
        self.assertAlmostEqual(0, track_in.dir.x)
        self.assertAlmostEqual(0, track_in.dir.y)
        self.assertAlmostEqual(1, track_in.dir.z)
        self.assertEqual(8, track_in.E)
        self.assertEqual(9, track_in.time)
        self.assertEqual(130, track_in.particle_type)  # this should be PDG!
        self.assertEqual(11, track_in.length)


class TestTrackFit(TestCase):

    def setUp(self):
        data = (1, 2., 3., 4., 0., 0., 1., 8, 9, 10, 11, 12, 13, 14)
        self.track_fit = TrackFit(data, zed_correction=0)

    def test_trackfit_init(self):
        track_fit = self.track_fit
        self.assertEqual(1, track_fit.id)
        self.assertAlmostEqual(2, track_fit.pos.x)
        self.assertAlmostEqual(3, track_fit.pos.y)
        self.assertAlmostEqual(4, track_fit.pos.z)
        self.assertAlmostEqual(0, track_fit.dir.x)
        self.assertAlmostEqual(0, track_fit.dir.y)
        self.assertAlmostEqual(1, track_fit.dir.z)
        self.assertEqual(8, track_fit.E)
        self.assertEqual(9, track_fit.time)
        self.assertEqual(10, track_fit.speed)
        self.assertEqual(11, track_fit.ts)
        self.assertEqual(12, track_fit.te)
        self.assertEqual(13, track_fit.con1)
        self.assertEqual(14, track_fit.con2)


class TestNeutrino(TestCase):

    def setUp(self):
        data = (1, 2., 3., 4., 0., 0., 1., 8, 9, 10, 11, 12, 13, 14)
        self.neutrino = Neutrino(data, zed_correction=0)

    def test_neutrino_init(self):
        neutrino = self.neutrino
        self.assertEqual(1, neutrino.id)
        self.assertAlmostEqual(2, neutrino.pos.x)
        self.assertAlmostEqual(3, neutrino.pos.y)
        self.assertAlmostEqual(4, neutrino.pos.z)
        self.assertAlmostEqual(0, neutrino.dir.x)
        self.assertAlmostEqual(0, neutrino.dir.y)
        self.assertAlmostEqual(1, neutrino.dir.z)
        self.assertEqual(8, neutrino.E)
        self.assertEqual(9, neutrino.time)
        self.assertEqual(10, neutrino.Bx)
        self.assertEqual(11, neutrino.By)
        self.assertEqual(12, neutrino.ichan)
        self.assertEqual(13, neutrino.particle_type)
        self.assertEqual(14, neutrino.channel)

    def test_neutrino_str(self):
        neutrino = self.neutrino
        repr_str = "Neutrino: mu-, 8.0 GeV, NC"
        self.assertEqual(repr_str, str(neutrino))
        neutrino.E = 2000
        repr_str = "Neutrino: mu-, 2.0 TeV, NC"
        self.assertEqual(repr_str, str(neutrino))
        neutrino.E = 3000000
        repr_str = "Neutrino: mu-, 3.0 PeV, NC"
        self.assertEqual(repr_str, str(neutrino))


class TestEvtHit(TestCase):

    def test_hit_init(self):
        hit = EvtHit(1, 2, 3, 4, 5, 6, 7, 8)
        self.assertEqual(1, hit.id)
        self.assertEqual(2, hit.pmt_id)
        self.assertEqual(3, hit.pe)
        self.assertEqual(4, hit.time)
        self.assertEqual(5, hit.type)
        self.assertEqual(6, hit.n_photons)
        self.assertEqual(7, hit.track_in)
        self.assertEqual(8, hit.c_time)

    def test_hit_default_values(self):
        hit = EvtHit()
        self.assertIsNone(hit.id)
        self.assertIsNone(hit.pmt_id)
        self.assertIsNone(hit.time)

    def test_hit_default_values_are_set_if_others_are_given(self):
        hit = EvtHit(track_in=1)
        self.assertIsNone(hit.id)
        self.assertIsNone(hit.time)

    def test_hit_attributes_are_immutable(self):
        hit = EvtHit(1, True)
        with self.assertRaises(AttributeError):
            hit.time = 10


class TestEvtRawHit(TestCase):

    def test_rawhit_init(self):
        raw_hit = EvtRawHit(1, 2, 3, 4)
        self.assertEqual(1, raw_hit.id)
        self.assertEqual(2, raw_hit.pmt_id)
        self.assertEqual(3, raw_hit.tot)
        self.assertEqual(4, raw_hit.time)

    def test_hit_default_values(self):
        raw_hit = EvtRawHit()
        self.assertIsNone(raw_hit.id)
        self.assertIsNone(raw_hit.pmt_id)
        self.assertIsNone(raw_hit.time)

    def test_hit_default_values_are_set_if_others_are_given(self):
        raw_hit = EvtRawHit(pmt_id=1)
        self.assertIsNone(raw_hit.id)
        self.assertIsNone(raw_hit.time)

    def test_hit_attributes_are_immutable(self):
        raw_hit = EvtRawHit(1, True)
        with self.assertRaises(AttributeError):
            raw_hit.time = 10

    def test_hit_addition_remains_time_id_and_pmt_id_and_adds_tot(self):
        hit1 = EvtRawHit(id=1, time=1, pmt_id=1, tot=10)
        hit2 = EvtRawHit(id=2, time=2, pmt_id=2, tot=20)
        merged_hit = hit1 + hit2
        self.assertEqual(1, merged_hit.id)
        self.assertEqual(1, merged_hit.time)
        self.assertEqual(1, merged_hit.pmt_id)
        self.assertEqual(30, merged_hit.tot)

    def test_hit_addition_picks_correct_time_if_second_hit_is_earlier(self):
        hit1 = EvtRawHit(id=1, time=2, pmt_id=1, tot=10)
        hit2 = EvtRawHit(id=2, time=1, pmt_id=2, tot=20)
        merged_hit = hit1 + hit2
        self.assertEqual(2, merged_hit.id)
        self.assertEqual(2, merged_hit.pmt_id)

    def test_hit_additions_works_with_multiple_hits(self):
        hit1 = EvtRawHit(id=1, time=2, pmt_id=1, tot=10)
        hit2 = EvtRawHit(id=2, time=1, pmt_id=2, tot=20)
        hit3 = EvtRawHit(id=3, time=1, pmt_id=3, tot=30)
        merged_hit = hit1 + hit2 + hit3
        self.assertEqual(2, merged_hit.pmt_id)
        self.assertEqual(60, merged_hit.tot)
        self.assertEqual(1, merged_hit.time)
        self.assertEqual(2, merged_hit.id)

    def test_hit_addition_works_with_sum(self):
        hit1 = EvtRawHit(id=1, time=2, pmt_id=1, tot=10)
        hit2 = EvtRawHit(id=2, time=1, pmt_id=2, tot=20)
        hit3 = EvtRawHit(id=3, time=1, pmt_id=3, tot=30)
        hits = [hit1, hit2, hit3]
        merged_hit = reduce(operator.add, hits)
        self.assertEqual(2, merged_hit.id)
        self.assertEqual(1, merged_hit.time)
        self.assertEqual(60, merged_hit.tot)
        self.assertEqual(2, merged_hit.pmt_id)
