#!/usr/bin/env python3
import tempfile
import unittest

from km3net_testdata import data_path

import km3pipe as kp
import km3modules as km
import numpy as np
import km3io
import awkward as ak


class TestOfflineHeaderTabulator(unittest.TestCase):
    def test_module(self):
        outfile = tempfile.NamedTemporaryFile(delete=True)

        pipe = kp.Pipeline()
        pipe.attach(kp.io.OfflinePump, filename=data_path("offline/numucc.root"))
        pipe.attach(km.io.OfflineHeaderTabulator)
        pipe.attach(kp.io.HDF5Sink, filename=outfile.name)
        pipe.drain()

        pipe = kp.Pipeline()
        pipe.attach(kp.io.HDF5Pump, filename=outfile.name)
        pipe.attach(km.common.Observer, count=10, required_keys=["RawHeader"])
        pipe.drain()


class TestEventInfoTabulator(unittest.TestCase):
    def test_module(self):
        outfile = tempfile.NamedTemporaryFile(delete=True)

        pipe = kp.Pipeline()
        pipe.attach(
            kp.io.OfflinePump,
            filename=data_path(
                "offline/mcv6.0.gsg_muon_highE-CC_50-500GeV.km3sim.jterbr00008357.jorcarec.aanet.905.root"
            ),
        )
        pipe.attach(km.io.EventInfoTabulator)
        pipe.attach(kp.io.HDF5Sink, filename=outfile.name)
        pipe.drain(10)

        pipe = kp.Pipeline()
        pipe.attach(kp.io.HDF5Pump, filename=outfile.name)
        pipe.attach(km.common.Observer, count=10, required_keys=["EventInfo"])
        pipe.attach(CheckW2listContents)
        pipe.drain()


class TestHitsTabulator(unittest.TestCase):
    def test_offline_hits(self):
        outfile = tempfile.NamedTemporaryFile(delete=True)

        pipe = kp.Pipeline()
        pipe.attach(kp.io.OfflinePump, filename=data_path("offline/numucc.root"))
        pipe.attach(km.io.HitsTabulator, kind="offline")
        pipe.attach(kp.io.HDF5Sink, filename=outfile.name)
        pipe.drain()

        pipe = kp.Pipeline()
        pipe.attach(kp.io.HDF5Pump, filename=outfile.name)
        pipe.attach(km.common.Observer, count=10, required_keys=["Hits"])
        pipe.drain()

    def test_mc_hits(self):
        outfile = tempfile.NamedTemporaryFile(delete=True)

        pipe = kp.Pipeline()
        pipe.attach(kp.io.OfflinePump, filename=data_path("offline/numucc.root"))
        pipe.attach(km.io.HitsTabulator, kind="mc")
        pipe.attach(kp.io.HDF5Sink, filename=outfile.name)
        pipe.drain()

        pipe = kp.Pipeline()
        pipe.attach(kp.io.HDF5Pump, filename=outfile.name)
        pipe.attach(km.common.Observer, count=10, required_keys=["McHits"])
        pipe.drain()


class TestMCTracksTabulator(unittest.TestCase):
    def test_module(self):
        outfile = tempfile.NamedTemporaryFile(delete=True)

        pipe = kp.Pipeline()
        pipe.attach(kp.io.OfflinePump, filename=data_path("offline/numucc.root"))
        pipe.attach(km.io.MCTracksTabulator)
        pipe.attach(kp.io.HDF5Sink, filename=outfile.name)
        pipe.drain()

        pipe = kp.Pipeline()
        pipe.attach(kp.io.HDF5Pump, filename=outfile.name)
        pipe.attach(km.common.Observer, count=10, required_keys=["McTracks"])
        pipe.drain()


class TestRecoTracksTabulator(unittest.TestCase):
    def test_module(self):
        outfile = tempfile.NamedTemporaryFile(delete=True)

        pipe = kp.Pipeline()
        pipe.attach(
            kp.io.OfflinePump,
            filename=data_path(
                "offline/mcv6.0.gsg_muon_highE-CC_50-500GeV.km3sim.jterbr00008357.jorcarec.aanet.905.root"
            ),
        )
        pipe.attach(km.io.RecoTracksTabulator, best_tracks=True)
        pipe.attach(kp.io.HDF5Sink, filename=outfile.name)
        pipe.drain(5)

        pipe = kp.Pipeline()
        pipe.attach(kp.io.HDF5Pump, filename=outfile.name)
        pipe.attach(km.common.Observer, count=5, required_keys=["Tracks"])
        pipe.attach(km.common.Observer, count=5, required_keys=["RecStages"])
        pipe.attach(km.common.Observer, count=5, required_keys=["BestJmuon"])
        pipe.attach(CheckRecoContents)
        pipe.drain()


class CheckRecoContents(kp.Module):
    def configure(self):

        # use this to count through the single events
        self.event_idx = 0

        # get the original file to compare to
        filename = data_path(
            "offline/mcv6.0.gsg_muon_highE-CC_50-500GeV.km3sim.jterbr00008357.jorcarec.aanet.905.root"
        )
        self.f = km3io.OfflineReader(filename)

    def process(self, blob):

        # first, get some extracted values form the h5 file

        # best track
        jmuon_dir_z = blob["BestJmuon"].dir_z  # a reco parameter
        jmuon_jgandalf_chi2 = blob["BestJmuon"].JGANDALF_CHI2  # a fitinf parameter

        # all tracks
        tracks_dir_z = blob["Tracks"].dir_z
        tracks_jgandalf_chi2 = blob["Tracks"].JGANDALF_CHI2

        # then, get the values from the original file

        # all tracks
        all_tracks_raw = self.f.events[self.event_idx].tracks
        all_tracks_raw_dir_z = all_tracks_raw.dir_z
        all_tracks_raw_fitinf = self._preprocess_fitinf(all_tracks_raw.fitinf)
        all_tracks_raw_jgandalf_chi2 = all_tracks_raw_fitinf[
            :, km3io.definitions.fitparameters.JGANDALF_CHI2
        ]

        # best tracks
        best_tracks_raw = km3io.tools.best_jmuon(all_tracks_raw)
        best_tracks_raw_dir_z = best_tracks_raw.dir_z
        best_tracks_raw_jgandalf_chi2 = best_tracks_raw.fitinf[
            km3io.definitions.fitparameters.JGANDALF_CHI2
        ]

        # and finally compare
        assert np.allclose(best_tracks_raw_dir_z, jmuon_dir_z)
        assert np.allclose(best_tracks_raw_jgandalf_chi2, jmuon_jgandalf_chi2)

        # since an ak array wise assertion is not possible, do this element wise
        for i in range(len(all_tracks_raw_dir_z)):

            assert np.allclose(all_tracks_raw_dir_z[i], tracks_dir_z[i])

            # exclude nans from the assertion as they are not detected as equal
            if not np.isnan(all_tracks_raw_jgandalf_chi2[i]) and not np.isnan(
                tracks_jgandalf_chi2[i]
            ):
                assert np.allclose(
                    all_tracks_raw_jgandalf_chi2[i], tracks_jgandalf_chi2[i]
                )

        self.event_idx += 1

        return blob

    def _preprocess_fitinf(self, fitinf):
        # preprocess the fitinf a bit - yay!
        n_columns = max(km3io.definitions.fitparameters.values()) + 1
        fitinf_array = np.ma.filled(
            ak.to_numpy(ak.pad_none(fitinf, target=n_columns, axis=-1)),
            fill_value=np.nan,
        ).astype("float32")
        return fitinf_array


class CheckW2listContents(kp.Module):
    def configure(self):

        # use this to count through the single events
        self.event_idx = 0

        # get the original file to compare to
        filename = data_path(
            "offline/mcv6.0.gsg_muon_highE-CC_50-500GeV.km3sim.jterbr00008357.jorcarec.aanet.905.root"
        )
        self.f = km3io.OfflineReader(filename)

    def process(self, blob):

        # extracted values
        by = blob["EventInfo"].W2LIST_GSEAGEN_BY[0]
        cc = blob["EventInfo"].W2LIST_GSEAGEN_CC[0]

        # original values
        original_by = self.f.events[self.event_idx].w2list[
            km3io.definitions.w2list_gseagen["W2LIST_GSEAGEN_BY"]
        ]
        original_cc = self.f.events[self.event_idx].w2list[
            km3io.definitions.w2list_gseagen["W2LIST_GSEAGEN_CC"]
        ]

        # and compare
        assert by == original_by
        assert cc == original_cc

        self.event_idx += 1

        return blob
