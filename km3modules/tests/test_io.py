#!/usr/bin/env python3
import tempfile
import unittest

from km3net_testdata import data_path

import km3pipe as kp
import km3modules as km


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
        pipe.attach(kp.io.OfflinePump, filename=data_path("offline/numucc.root"))
        pipe.attach(km.io.EventInfoTabulator)
        pipe.attach(kp.io.HDF5Sink, filename=outfile.name)
        pipe.drain()

        pipe = kp.Pipeline()
        pipe.attach(kp.io.HDF5Pump, filename=outfile.name)
        pipe.attach(km.common.Observer, count=10, required_keys=["EventInfo"])
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
                "offline/mcv5.11r2.gsg_muonCChigherE-CC_50-5000GeV.km3_AAv1.jterbr00004695.jchain.aanet.498.root"
            ),
        )
        pipe.attach(km.io.RecoTracksTabulator)
        pipe.attach(kp.io.HDF5Sink, filename=outfile.name)
        pipe.drain(5)

        pipe = kp.Pipeline()
        pipe.attach(kp.io.HDF5Pump, filename=outfile.name)
        pipe.attach(km.common.Observer, count=5, required_keys=["Tracks"])
        pipe.attach(km.common.Observer, count=5, required_keys=["RecStages"])
        pipe.drain()
