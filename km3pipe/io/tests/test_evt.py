# Filename: test_evt.py
# pylint: disable=locally-disabled,C0111,R0904,C0301,C0103,W0212

from io import StringIO
from os.path import join, dirname

import numpy as np

from km3pipe.testing import TestCase, skip, data_path
from km3pipe.io.evt import EvtPump, EVT_PARSERS

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestEvtPump(TestCase):
    def setUp(self):
        self.valid_evt_header = "\n".join(
            (
                "start_run: 1",
                "cut_nu: 0.100E+03 0.100E+09-0.100E+01 0.100E+01",
                "spectrum: -1.40",
                "physics: gSeaGen 4.1 180126  165142",
                "physics: GENIE 2.10.2 180126  165142",
                "end_event:",
                "start_event: 12 1",
                "track_in: 1 -389.951 213.427 516 -0.204562 -0.60399 -0.770293"
                + " 9.092 0 5 40.998",
                "hit: 1 44675 1 1170.59 5 2 1 1170.59",
                "end_event:",
                "start_event: 13 1",
                "track_in:  1 272.695 -105.613 516 -0.425451 -0.370522 -0.825654"
                + " 2431.47 0 5 -1380",
                "track_in:  2 272.348 -106.292 516 -0.425451 -0.370522 -0.825654"
                + " 24670.7 1.33 5 -1484",
                "track_in:  3 279.47 -134.999 516 -0.425451 -0.370522 -0.825654"
                + " 164.586 26.7 5 601.939",
                "hit: 1 20140 1 1140.06 5 1 1 1140.06",
                "hit: 2 20159 1 1177.14 5 1 1 1177.14",
                "hit: 3 20164 1 1178.19 5 1 1 1178.19",
                "hit: 4 20170 1 1177.23 5 1 1 1177.23",
                "hit: 5 20171 2 1177.25 5 1 2 1177.25",
                "end_event:",
                "start_event: 14 1",
                "track_in:  1 40.256 -639.888 516 0.185998 0.476123 -0.859483"
                + " 10016.1 0 5 -1668",
                "hit: 1 33788 1 2202.81 5 1 1 2202.81",
                "hit: 2 33801 1 2248.95 5 1 1 2248.95",
                "hit: 3 33814 1 2249.2 5 1 1 2249.2",
                "end_event:",
            )
        )
        self.no_evt_header = "\n".join(
            (
                "start_event: 12 1",
                "track_in: 1 -389.951 213.427 516 -0.204562 -0.60399 -0.770293"
                + " 9.092 0 5 40.998",
                "hit: 1 44675 1 1170.59 5 2 1 1170.59",
                "end_event:",
                "start_event: 13 1",
                "track_in:  1 272.695 -105.613 516 -0.425451 -0.370522 -0.825654"
                + " 2431.47 0 5 -1380",
                "track_in:  2 272.348 -106.292 516 -0.425451 -0.370522 -0.825654"
                + " 24670.7 1.33 5 -1484",
                "track_in:  3 279.47 -134.999 516 -0.425451 -0.370522 -0.825654"
                + " 164.586 26.7 5 601.939",
                "hit: 1 20140 1 1140.06 5 1 1 1140.06",
                "hit: 2 20159 1 1177.14 5 1 1 1177.14",
                "hit: 3 20164 1 1178.19 5 1 1 1178.19",
                "hit: 4 20170 1 1177.23 5 1 1 1177.23",
                "hit: 5 20171 2 1177.25 5 1 2 1177.25",
                "end_event:",
                "start_event: 14 1",
                "track_in:  1 40.256 -639.888 516 0.185998 0.476123 -0.859483"
                + " 10016.1 0 5 -1668",
                "hit: 1 33788 1 2202.81 5 1 1 2202.81",
                "hit: 2 33801 1 2248.95 5 1 1 2248.95",
                "hit: 3 33814 1 2249.2 5 1 1 2249.2",
                "end_event:",
            )
        )
        self.corrupt_evt_header = "foo"
        self.corrupt_line = "\n".join(
            ("start_event: 1 1", "corrupt line", "end_event:")
        )

        self.pump = EvtPump(parsers=[])
        self.pump.blob_file = StringIO(self.valid_evt_header)

    def tearDown(self):
        self.pump.blob_file.close()

    def test_parse_header(self):
        raw_header = self.pump.extract_header()
        self.assertEqual([["1"]], raw_header["start_run"])
        self.assertAlmostEqual(-1.4, float(raw_header["spectrum"][0][0]))
        self.assertAlmostEqual(1, float(raw_header["cut_nu"][0][2]))

    def test_header_entries_with_same_tag_are_put_in_lists(self):
        raw_header = self.pump.extract_header()
        self.assertAlmostEqual(2, len(raw_header["physics"]))
        self.assertAlmostEqual(1, len(raw_header["spectrum"]))
        assert "gSeaGen" == raw_header["physics"][0][0]
        assert "GENIE" == raw_header["physics"][1][0]

    #    def test_incomplete_header_raises_value_error(self):
    #        temp_file = StringIO(self.corrupt_evt_header)
    #        pump = EvtPump()
    #        pump.blob_file = temp_file
    #        with self.assertRaises(ValueError):
    #            pump.extract_header()
    #        temp_file.close()

    def test_record_offset_saves_correct_offset(self):
        self.pump.blob_file = StringIO("a" * 42)
        offsets = [1, 4, 9, 12, 23]
        for offset in offsets:
            self.pump.blob_file.seek(0, 0)
            self.pump.blob_file.seek(offset, 0)
            self.pump._record_offset()
        self.assertListEqual(offsets, self.pump.event_offsets)

    def test_event_offset_is_at_first_event_after_parsing_header(self):
        self.pump.extract_header()
        self.assertEqual(161, self.pump.event_offsets[0])

    def test_rebuild_offsets(self):
        self.pump.extract_header()
        self.pump._cache_offsets()
        self.assertListEqual([161, 306, 773], self.pump.event_offsets)

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
        self.assertTupleEqual((14, 1), blob["start_event"])
        self.assertEqual(3, len(self.pump.event_offsets))

    def test_get_blob_raises_index_error_for_wrong_index(self):
        self.pump.prepare_blobs()
        with self.assertRaises(IndexError):
            self.pump.get_blob(23)

    def test_get_blob_returns_correct_event_information(self):
        self.pump.prepare_blobs()
        blob = self.pump.get_blob(0)
        self.assertTrue("raw_header" in blob)
        self.assertEqual([["1"]], blob["raw_header"]["start_run"])
        self.assertTupleEqual((12, 1), blob["start_event"])
        # TODO: all the other stuff like hit, track etc.
        assert "hit" in blob
        assert "track_in" in blob
        assert np.allclose(
            [[1.0, 44675.0, 1.0, 1170.59, 5.0, 2.0, 1.0, 1170.59]], blob["hit"]
        )
        blob = self.pump.get_blob(1)
        assert 5 == len(blob["hit"])
        assert np.allclose(
            [3.0, 20164.0, 1.0, 1178.19, 5.0, 1.0, 1.0, 1178.19], blob["hit"][2]
        )

    def test_get_blob_returns_correct_events(self):
        self.pump.prepare_blobs()
        blob = self.pump.get_blob(0)
        self.assertTupleEqual((12, 1), blob["start_event"])
        blob = self.pump.get_blob(2)
        self.assertTupleEqual((14, 1), blob["start_event"])
        blob = self.pump.get_blob(1)
        self.assertTupleEqual((13, 1), blob["start_event"])

    def test_process_returns_correct_blobs(self):
        self.pump.prepare_blobs()
        blob = self.pump.process()
        self.assertTupleEqual((12, 1), blob["start_event"])
        blob = self.pump.process()
        self.assertTupleEqual((13, 1), blob["start_event"])
        blob = self.pump.process()
        self.assertTupleEqual((14, 1), blob["start_event"])

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
            event_numbers.append(blob["start_event"][0])
        self.assertListEqual([12, 13, 14], event_numbers)

    def test_pump_has_len(self):
        self.pump.prepare_blobs()
        self.assertEqual(3, len(self.pump))

    def test_pump_get_item_returns_first_for_index_zero(self):
        self.pump.prepare_blobs()
        first_blob = self.pump[0]
        self.assertEqual(12, first_blob["start_event"][0])

    def test_pump_get_item_returns_correct_blob_for_index(self):
        self.pump.prepare_blobs()
        blob = self.pump[1]
        self.assertEqual(13, blob["start_event"][0])

    def test_pump_slice_generator(self):
        self.pump.prepare_blobs()
        blobs = self.pump[:]
        blobs = list(self.pump[1:3])
        self.assertEqual(2, len(blobs))
        self.assertEqual((13, 1), blobs[0]["start_event"])

    def test_create_blob_entry_for_line_ignores_corrupt_line(self):
        self.pump.blob_file = StringIO(self.corrupt_line)
        self.pump.extract_header()
        self.pump.prepare_blobs()
        self.pump.get_blob(0)

    def test_parsers_are_ignored_if_not_valid(self):
        self.pump = EvtPump(parsers=["a", "b"])
        self.pump.blob_file = StringIO(self.valid_evt_header)
        assert "a" not in self.pump.parsers
        assert "b" not in self.pump.parsers

    def test_parsers_are_added(self):
        self.pump = EvtPump(parsers=["km3sim"])
        self.pump.blob_file = StringIO(self.valid_evt_header)
        assert EVT_PARSERS["km3sim"] in self.pump.parsers

    def test_custom_parser(self):
        def a_parser(blob):
            blob["foo"] = 23

        self.pump = EvtPump(parsers=[a_parser])
        self.pump.blob_file = StringIO(self.valid_evt_header)
        self.pump.extract_header()
        self.pump.prepare_blobs()
        blob = self.pump[0]

        assert 23 == blob["foo"]

    def test_auto_parser_finds_all_physics_parsers(self):
        self.pump = EvtPump(parsers="auto")
        self.pump.blob_file = StringIO(self.valid_evt_header)
        self.pump.extract_header()
        assert EVT_PARSERS["gseagen"] in self.pump.parsers


class TestEvtFilePump(TestCase):
    def setUp(self):
        self.fname = data_path("evt/example_numuNC.evt")

    def test_pipe(self):
        pump = EvtPump(filename=self.fname)
        next(pump)
        pump.finish()


class TestCorsika(TestCase):
    def setUp(self):
        self.fname = data_path("evt/example_corant_propa.evt")

    def test_pipe(self):
        pump = EvtPump(filename=self.fname)
        next(pump)
        pump.finish()


class TestPropa(TestCase):
    def setUp(self):
        self.fname = data_path("evt/example_corant_propa.evt")
        self.fnames = []
        for i in [0, 1]:
            self.fnames.append(data_path("evt/example_corant_propa.evt"))

    def test_pipe(self):
        pump = EvtPump(filename=self.fname, parsers=["propa"])
        assert EVT_PARSERS["propa"] in pump.parsers
        blob = next(pump)
        assert "start_event" in blob
        assert "track_primary" in blob
        assert "track_in" in blob
        pump.finish()

    def test_filenames(self):
        pump = EvtPump(filenames=self.fnames, parsers=["propa"])
        assert EVT_PARSERS["propa"] in pump.parsers
        blob = next(pump)
        assert "start_event" in blob
        assert "track_primary" in blob
        assert "track_in" in blob
        pump.finish()

    @skip
    def test_auto_parser(self):
        pump = EvtPump(filename=self.fname)
        assert EVT_PARSERS["propa"] in pump.parsers
        blob = next(pump)
        assert "start_event" in blob
        assert "track_primary" in blob
        assert "Muon" in blob
        assert "MuonMultiplicity" in blob
        assert "Neutrino" in blob
        assert "NeutrinoMultiplicity" in blob
        assert "Weights" in blob
        assert "Primary" in blob
        pump.finish()


class TestKM3Sim(TestCase):
    def setUp(self):
        self.fname = data_path("evt/KM3Sim.evt")

    def test_pipe(self):
        pump = EvtPump(filename=self.fname, parsers=["km3sim"])
        assert EVT_PARSERS["km3sim"] in pump.parsers
        next(pump)
        pump.finish()

    def test_hits(self):
        pump = EvtPump(filename=self.fname, parsers=["km3sim"])
        blob = pump[0]
        hits = blob["KM3SimHits"]
        assert 4 == len(hits)
        assert 195749 == hits[0].pmt_id

    def test_neutrino(self):
        pump = EvtPump(filename=self.fname, parsers=["gseagen", "km3sim"])
        blob = pump[0]
        EVT_PARSERS["gseagen"](blob)
        neutrino = blob["Neutrinos"][0]
        self.assertAlmostEqual(0.10066, neutrino.energy)


class TestParserDetection(TestCase):
    def test_parsers_are_automatically_detected(self):
        pass
