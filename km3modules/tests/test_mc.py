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

from km3modules.mc import convert_mctime, MCTimeCorrector

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2018, Tamas Gal and the KM3NeT collaboration."
__license__ = "MIT"
__maintainer__ = "Tamas Gal, Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestMCConvert(TestCase):
    def setUp(self):
        self.info = Table({
            'timestamp': 12.3,
            'nanoseconds': 42,
            'mc_time': 10000,
        })
        self.hits = Table({
            'time': 874.42,
        })
        self.blob = Blob({
            'Info': self.info,
            'Les_Hits': self.hits,
        })

    def test_convert(self):
        times = convert_mctime(
            self.hits, self.info.timestamp, self.info.nanoseconds
        )
        assert times is not None

    def test_process(self):
        corr = MCTimeCorrector(hits_key='Les_Hits', event_info_key='Info')
        newblob = corr.process(self.blob)
        assert newblob['Les_Hits'] is not None

    def test_pipe(self):
        p = Pipeline()
        pass
