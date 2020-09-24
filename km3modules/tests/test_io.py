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

class TestEventInfoTabulator(unittest.TestCase):
    def test_module(self):
        outfile = tempfile.NamedTemporaryFile(delete=True)

        pipe = kp.Pipeline()
        pipe.attach(kp.io.OfflinePump, filename=data_path("offline/numucc.root"))
        pipe.attach(km.io.EventInfoTabulator)
        pipe.attach(kp.io.HDF5Sink, filename=outfile.name)
        pipe.drain()


class TestHitsTabulator(unittest.TestCase):
    def test_offline_hits(self):
        outfile = tempfile.NamedTemporaryFile(delete=True)

        pipe = kp.Pipeline()
        pipe.attach(kp.io.OfflinePump, filename=data_path("offline/numucc.root"))
        pipe.attach(km.io.HitsTabulator, kind="offline")
        pipe.attach(kp.io.HDF5Sink, filename=outfile.name)
        pipe.drain()

    def test_mc_hits(self):
        outfile = tempfile.NamedTemporaryFile(delete=True)

        pipe = kp.Pipeline()
        pipe.attach(kp.io.OfflinePump, filename=data_path("offline/numucc.root"))
        pipe.attach(km.io.HitsTabulator, kind="mc")
        pipe.attach(kp.io.HDF5Sink, filename=outfile.name)
        pipe.drain()

class TestMCTracksTabulator(unittest.TestCase):
    def test_module(self):
        outfile = tempfile.NamedTemporaryFile(delete=True)

        pipe = kp.Pipeline()
        pipe.attach(kp.io.OfflinePump, filename=data_path("offline/numucc.root"))
        pipe.attach(km.io.MCTracksTabulator)
        pipe.attach(kp.io.HDF5Sink, filename=outfile.name)
        pipe.drain()
