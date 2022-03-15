# Filename: test_mc.py
# pylint: disable=C0111,R0904,C0103
# vim:set ts=4 sts=4 sw=4 et:
"""
Tests for MC Modules.
"""

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest

from km3pipe import Table, Blob, Pipeline, Module
from km3pipe.testing import TestCase

from km3modules.mc import MCTimeCorrector, GlobalRandomState

__author__ = "Moritz Lotze, Michael Moser"
__copyright__ = "Copyright 2018, Tamas Gal and the KM3NeT collaboration."
__license__ = "MIT"
__maintainer__ = "Tamas Gal, Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestGlobalRandomState(TestCase):
    def test_default_random_state(self):
        assertAlmostEqual = self.assertAlmostEqual

        class Observer(Module):
            def configure(self):
                self.i = 0
                self.x = [0.3745401188, 0.950714306, 0.7319939418]

            def process(self, blob):
                assertAlmostEqual(self.x[self.i], np.random.rand())
                self.i += 1
                return blob

        pipe = Pipeline()
        pipe.attach(GlobalRandomState)
        pipe.attach(Observer)
        pipe.drain(3)

    def test_custom_random_state(self):
        assertAlmostEqual = self.assertAlmostEqual

        class Observer(Module):
            def configure(self):
                self.i = 0
                self.x = [0.221993171, 0.870732306, 0.206719155]

            def process(self, blob):
                assertAlmostEqual(self.x[self.i], np.random.rand())
                self.i += 1
                return blob

        pipe = Pipeline()
        pipe.attach(GlobalRandomState, seed=5)
        pipe.attach(Observer)
        pipe.drain(3)

    def test_without_pipeline_and_default_state(self):
        GlobalRandomState()
        numbers = np.arange(1, 50)
        np.random.shuffle(numbers)
        lotto_numbers = sorted(numbers[:6])
        self.assertListEqual([14, 18, 28, 45, 46, 48], lotto_numbers)

    def test_without_pipeline_with_custom_seed(self):
        GlobalRandomState(seed=23)
        numbers = np.arange(1, 50)
        np.random.shuffle(numbers)
        lotto_numbers = sorted(numbers[:6])
        self.assertListEqual([14, 15, 18, 19, 33, 44], lotto_numbers)


class TestMCConvert(TestCase):
    def setUp(self):
        self.event_info = Table(
            {
                "timestamp": 1,
                "nanoseconds": 700000000,
                "mc_time": 1.74999978e9,
            }
        )

        self.mc_tracks = Table(
            {
                "time": 1,
            }
        )

        self.mc_hits = Table(
            {
                "time": 30.79,
            }
        )

        self.blob = Blob(
            {
                "event_info": self.event_info,
                "mc_hits": self.mc_hits,
                "mc_tracks": self.mc_tracks,
            }
        )

    def test_process(self):
        corr = MCTimeCorrector(
            mc_hits_key="mc_hits",
            mc_tracks_key="mc_tracks",
            event_info_key="event_info",
        )
        newblob = corr.process(self.blob)
        assert newblob["mc_hits"] is not None
        assert newblob["mc_tracks"] is not None
        assert np.allclose(newblob["mc_hits"].time, 49999810.79)
        assert np.allclose(newblob["mc_tracks"].time, 49999781)
