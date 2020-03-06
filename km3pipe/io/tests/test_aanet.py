# Filename: test_aanet.py
# pylint: disable=locally-disabled,C0111,R0904,C0301,C0103,W0212
from os.path import join, dirname
import time
import numpy as np

import km3pipe as kp
from km3pipe.testing import TestCase, patch, Mock, skipif, skip
from km3pipe.io.aanet import OfflinePump, MetaParser
from km3pipe.core import Pipeline, Module, Pump

try:
    import aa
    NO_AANET = False
except ImportError:
    import sys
    sys.modules['ROOT'] = Mock()
    sys.modules['aa'] = Mock()
    NO_AANET = True
else:
    import ROOT

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2018, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

TEST_DATA_DIR = join(dirname(__file__), "../../kp-data/test_data")


class TestOfflinePump(TestCase):
    def setUp(self):
        self.fname = join(TEST_DATA_DIR, 'mupage.root')
        self.pump = OfflinePump(filename=self.fname)

    def test_reading_hits(self):
        data = {
            'hit_lengths': [399, 779, 12],
            'mean_times': [2130.1878, 3252.8879, 2082.5918],
            'pmt_id_sums': [36668781, 16629383, 1433076]
        }

        blob_counter = 0
        for idx, blob in enumerate(self.pump):
            blob_counter += 1
            hits = blob['McHits']
            assert len(hits) == data['hit_lengths'][idx]
            self.assertAlmostEqual(
                data['mean_times'][idx], np.mean(hits.time), 3
            )
            assert data['pmt_id_sums'][idx] == np.sum(hits.pmt_id)
        assert 3 == blob_counter

    def test_event_info(self):
        data = {
            'run_ids': [0] * 3,
            'event_ids': [0, 0, 0],
            'frame_indices': [0] * 3,
            'mc_ids': [4, 8, 9],
            'timestamps': [1545418638, 1545418639, 1545418639],
            'nanoseconds': [922485000, 148073000, 299241000],
            'trigger_counters': [0, 0, 0],
            'trigger_masks': [0, 0, 0],
            'overlays': [0, 0, 0],
        }
        blob_counter = 0
        for idx, blob in enumerate(self.pump):
            blob_counter += 1
            ei = blob['EventInfo'][0]
            assert data['run_ids'][idx] == ei.run_id
            assert data['event_ids'][idx] == ei.event_id
            assert data['frame_indices'][idx] == ei.frame_index
            assert data['timestamps'][idx] == ei.timestamp
            assert data['trigger_counters'][idx] == ei.trigger_counter
            assert data['overlays'][idx] == ei.overlays
            assert data['trigger_masks'][idx] == ei.trigger_mask
            assert data['nanoseconds'][idx] == ei.nanoseconds
        assert 3 == blob_counter

    def test_aanet_and_hdf5_conformity_with_converted_files(self):
        aanet_pump = OfflinePump(filename=join(TEST_DATA_DIR, 'mupage.root'))
        hdf5_pump = kp.io.HDF5Pump(
            filename=join(TEST_DATA_DIR, 'mupage.root.h5')
        )

        blob_counter = 0
        for aanet_blob, hdf5_blob in zip(aanet_pump, hdf5_pump):
            blob_counter += 1
            keys = ['McHits', 'McTracks', 'EventInfo']
            for key in keys:
                aanet_data = aanet_blob[key]
                hdf5_data = hdf5_blob[key]
                assert aanet_data.dtype == hdf5_data.dtype

                for attr in aanet_data.dtype.names:
                    # special case for arrays which are all NaNs
                    if np.all(np.isnan(aanet_data[attr])):
                        assert np.all(np.isnan(hdf5_data[attr]))
                        continue
                    # otherwise just compare them
                    assert np.allclose(aanet_data[attr], hdf5_data[attr])

            # aanet_header = aanet_blob['Header']
            # hdf5_header = hdf5_blob['Header']
            # self.assertTupleEqual(aanet_header.seed, hdf5_header.seed)
            # self.assertTupleEqual(aanet_header.norma, hdf5_header.norma)
            # self.assertTupleEqual(
            #     aanet_header.coord_origin, hdf5_header.coord_origin
            # )
            # self.assertTupleEqual(aanet_header.physics, hdf5_header.physics)
            # self.assertTupleEqual(aanet_header.propag, hdf5_header.propag)
            # self.assertTupleEqual(aanet_header.livetime, hdf5_header.livetime)
            # self.assertTupleEqual(aanet_header.cut_in, hdf5_header.cut_in)
            # self.assertTupleEqual(aanet_header.can, hdf5_header.can)
            # self.assertTupleEqual(
            #     aanet_header.seabottom, hdf5_header.seabottom
            # )
            # self.assertTupleEqual(
            #     aanet_header.start_run, hdf5_header.start_run
            # )
            # self.assertTupleEqual(aanet_header.simul, hdf5_header.simul)
        assert 3 == blob_counter

    def test_against_single_file_freeze(self):
        start_time = time.time()

        aanet_pump = OfflinePump(filename=join(TEST_DATA_DIR, 'mupage.root'))
        assert 10 > (time.time() - start_time)


class TestMetaParser(TestCase):
    def test_init(self):
        MetaParser()

    def test_parse_string_of_single_entry(self):
        string = b"A 123\nA 1.2.3\nA KM3NET\nA a\nA b\nA Linux"
        mp = MetaParser()
        mp.parse_string(string)

        assert 1 == len(mp.meta)

        assert b'A' == mp.meta[0]['application_name']
        assert b'123' == mp.meta[0]['revision']
        assert b'A a\nA b' == mp.meta[0]['command']

    def test_parse_string_of_multiple_entries(self):
        string = (
            b"A 123\nA 1.2.3\nA KM3NET\nA a\nA b\nA Linux\n"
            b"B 456\nB 4.5.6\nB KM3NET\nB c\nB Linux"
        )
        mp = MetaParser()
        mp.parse_string(string)

        assert 2 == len(mp.meta)

        assert b'A' == mp.meta[0]['application_name']
        assert b'123' == mp.meta[0]['revision']
        assert b'1.2.3' == mp.meta[0]['root_version']
        assert b'A a\nA b' == mp.meta[0]['command']

        assert b'B' == mp.meta[1]['application_name']
        assert b'456' == mp.meta[1]['revision']
        assert b'4.5.6' == mp.meta[1]['root_version']
        assert b'B c' == mp.meta[1]['command']

    def test_parse_testfile(self):
        fname = join(TEST_DATA_DIR, 'jprintmeta.log')
        with open(fname, 'rb') as fobj:
            string = fobj.read()
        mp = MetaParser()
        mp.parse_string(string)
        assert 7 == len(mp.meta)

        assert b'JEvt' == mp.meta[0]['application_name']
        assert b'9912' == mp.meta[0]['revision']
        assert b'5.34/23' == mp.meta[0]['root_version']
        assert mp.meta[0]['command'].startswith(b'JEvt /pbs/throng/km3net')
        assert mp.meta[0]['command'].endswith(b'2 --!')

        assert b'JEnergy' == mp.meta[1]['application_name']
        assert b'9912' == mp.meta[1]['revision']
        assert b'5.34/23' == mp.meta[1]['root_version']
        assert mp.meta[1]['command'].startswith(b'JEnergy /pbs/throng/km3net')
        assert mp.meta[1]['command'].endswith(b'1 --!')

        assert b'JTriggerEfficiency' == mp.meta[-1]['application_name']
        assert b'8519' == mp.meta[-1]['revision']
        assert b'5.34/23' == mp.meta[-1]['root_version']
        assert mp.meta[-1]['command'].startswith(
            b'JTriggerEfficiency /pbs/throng/km3net'
        )
        assert mp.meta[-1]['command'].endswith(b'326 --!')

    def test_get_table(self):
        string = (
            b"A 123\nA 1.2.3\nA KM3NET\nA a\nA b\nA Linux\n"
            b"B 456\nB 4.5.6\nB KM3NET\nB c\nB Linux"
        )
        mp = MetaParser()
        mp.parse_string(string)
        tab = mp.get_table()
        assert 2 == len(tab)
        assert '/meta' == tab.h5loc
        assert tab.h5singleton
        assert b'B c' == tab['command'][1]
        assert np.dtype('S7') == tab['command'].dtype

    def test_empty_string(self):
        mp = MetaParser()
        mp.parse_string('')
        assert mp.get_table() is None
