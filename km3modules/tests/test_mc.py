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

from km3modules.mc import convert_hits_jte_t_to_mc_t
from km3modules.mc import convert_tracks_mc_t_to_jte_t, MCTimeCorrector

__author__ = "Moritz Lotze, Michael Moser"
__copyright__ = "Copyright 2018, Tamas Gal and the KM3NeT collaboration."
__license__ = "MIT"
__maintainer__ = "Tamas Gal, Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestMCConvert(TestCase):
    def setUp(self):
        self.event_info = Table({
            'timestamp': 12.3 * 1e-6,
            'nanoseconds': 42,
            'mc_time': 10000, })
            
        self.mc_tracks = Table({
            'time': 42, })
            
        self.hits = Table({
            'time': 874.42, })
            
        self.blob = Blob({
            'event_info': self.event_info,
            'hits': self.hits, })

    def test_convert_hits_jte_t_to_mc_t(self):
        times = convert_hits_jte_t_to_mc_t(
            self.hits.time, self.event_info.timestamp * 1e9 + 
            self.event_info.nanoseconds, self.event_info.mc_time)
            
        assert times is not None
        print(times)
        assert np.allclose(times, 3216.42)
        
    def test_convert_tracks_mc_t_to_jte_t(self):
        times = convert_tracks_mc_t_to_jte_t(
            self.mc_tracks.time, self.event_info.timestamp * 1e9 + 
            self.event_info.nanoseconds, self.event_info.mc_time)
            
        assert times is not None
        print(times)
        assert np.allclose(times, 3216.42) # TODO fix number

    def test_process(self):
        corr = MCTimeCorrector(True, False, 
                               hits_key='hits', event_info_key='event_info)
        newblob = corr.process(self.blob)
        assert newblob['hits'] is not None
        assert np.allclose(newblob['hits'].time, 3216.42) # TODO fix number
