import numpy as np
import km3net_testdata
import tempfile
import h5py
import km3pipe as kp
from km3pipe.utils.h5extractf import h5extractf
from km3pipe.testing import TestCase


class TestJsireneH5File(TestCase):
    @classmethod
    def setUpClass(cls):
        filename = km3net_testdata.data_path(
            "offline/mcv5.0.DAT004340.propa.sirene.jte.jchain.aanet.4340.root"
        )
        cls.h5file = tempfile.NamedTemporaryFile()
        h5extractf(filename, outfile=cls.h5file)

    @classmethod
    def tearDownClass(cls):
        cls.h5file.close()

    def test_datasets_names_and_lengths(self):
        target = {
            "event_info": 3,
            "group_info": 3,
            "hits": 280,
            "mc_hits": 598,
            "mc_tracks": 48,
            "raw_header": 21,
            "reco": 2,
        }
        with h5py.File(self.h5file, "r") as f:
            self.assertDictEqual(target, {x: len(f[x]) for x in f})

    def test_reco_datasets_names_and_lengths(self):
        target = {"best_jmuon": 3, "tracks": 76}
        with h5py.File(self.h5file, "r") as f:
            self.assertDictEqual(target, {x: len(f["reco"][x]) for x in f["reco"]})

    def test_time_of_first_three_hits(self):
        with h5py.File(self.h5file, "r") as f:
            np.testing.assert_array_equal(
                f["hits"]["time"][:3],
                [87530475.0, 87526107.0, 87527368.0],
            )

    def test_time_of_last_three_hits(self):
        with h5py.File(self.h5file, "r") as f:
            np.testing.assert_array_equal(
                f["hits"]["time"][-3:],
                [8166085.0, 8165582.0, 8163487.0],
            )

    def test_hits_group_ids(self):
        target = [0] * 86 + [1] * 111 + [2] * 83
        with h5py.File(self.h5file, "r") as f:
            np.testing.assert_array_equal(
                f["hits"]["group_id"],
                target,
            )

    def test_hits_dtype_names(self):
        target = (
            "channel_id",
            "dom_id",
            "time",
            "tot",
            "triggered",
            "pos_x",
            "pos_y",
            "pos_z",
            "dir_x",
            "dir_y",
            "dir_z",
            "tdc",
            "group_id",
        )
        with h5py.File(self.h5file, "r") as f:
            self.assertTupleEqual(
                f["hits"].dtype.names,
                target,
            )

    def test_h5_file_can_be_opened_with_hdf5pump_and_keys_are_correct(self):
        pump = kp.io.HDF5Pump(filename=self.h5file.name)
        blob = pump[1]
        target = {
            "BestJmuon",
            "EventInfo",
            "GroupInfo",
            "Header",
            "Hits",
            "McHits",
            "McTracks",
            "RawHeader",
            "Tracks",
        }
        self.assertSetEqual(target, set(blob.keys()))
