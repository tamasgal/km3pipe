# Filename: test_mc.py
# pylint: disable=C0111,R0904,C0103
# vim:set ts=4 sts=4 sw=4 et:
"""
Tests for MC Modules.
"""

import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose)
import pytest

from km3pipe import Table, Blob, Pipeline
from km3pipe.testing import TestCase

from km3modules.mc import convert_mc_times_to_jte_times, MCTimeCorrector

__author__ = "Moritz Lotze, Michael Moser"
__copyright__ = "Copyright 2018, Tamas Gal and the KM3NeT collaboration."
__license__ = "MIT"
__maintainer__ = "Tamas Gal, Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestMCConvert(TestCase):
    def setUp(self):
        self.event_info = Table({
            'timestamp': 1,
            'nanoseconds': 700000000,
            'mc_time': 1.74999978e9,
        })

        self.mc_tracks = Table({
            'time': 1,
        })

        self.mc_hits = Table({
            'time': 30.79,
        })

        self.blob = Blob({
            'event_info': self.event_info,
            'mc_hits': self.mc_hits,
            'mc_tracks': self.mc_tracks,
        })

    def test_convert_mc_times_to_jte_times(self):
        times_mc_tracks = convert_mc_times_to_jte_times(
            self.mc_tracks.time,
            self.event_info.timestamp * 1e9 + self.event_info.nanoseconds,
            self.event_info.mc_time
        )
        times_mc_hits = convert_mc_times_to_jte_times(
            self.mc_hits.time,
            self.event_info.timestamp * 1e9 + self.event_info.nanoseconds,
            self.event_info.mc_time
        )

        assert times_mc_tracks is not None
        assert times_mc_hits is not None
        print(times_mc_tracks, times_mc_hits)
        assert np.allclose(times_mc_tracks, 49999781)
        assert np.allclose(times_mc_hits, 49999810.79)

    def test_process(self):
        corr = MCTimeCorrector(
            mc_hits_key='mc_hits',
            mc_tracks_key='mc_tracks',
            event_info_key='event_info'
        )
        newblob = corr.process(self.blob)
        assert newblob['mc_hits'] is not None
        assert newblob['mc_tracks'] is not None
        assert np.allclose(newblob['mc_hits'].time, 49999810.79)
        assert np.allclose(newblob['mc_tracks'].time, 49999781)
