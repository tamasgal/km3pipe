# coding=utf-8
# Filename: test_hits.py
# pylint: disable=locally-disabled,C0111,R0904,C0103
from __future__ import division, absolute_import, print_function

import numpy as np
from pandas.util.testing import assert_frame_equal

from km3pipe.testing import TestCase
from km3pipe.dataclasses import HitSeries
from km3modules.hits import (HitSelector, HitStatistics, FirstHits, # noqa
                             NDoms, TrimmedHits) # noqa

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "BSD-3"
__maintainer__ = "Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"


class TestSelector(TestCase):
    def setUp(self):
        n = 10
        ids = np.arange(n)
        dom_ids = np.arange(n)
        dir_xs = np.arange(n)
        dir_ys = np.arange(n)
        dir_zs = np.arange(n)
        pos_xs = np.arange(n)
        pos_ys = np.arange(n)
        pos_zs = np.arange(n)
        t0s = np.arange(n)
        times = np.arange(n)
        tots = np.arange(n)
        channel_ids = np.arange(n)
        triggereds = np.ones(n)
        pmt_ids = np.arange(n)

        self.hits = HitSeries.from_arrays(
            channel_ids,
            dir_xs,
            dir_ys,
            dir_zs,
            dom_ids,
            ids,
            pos_xs,
            pos_ys,
            pos_zs,
            pmt_ids,
            t0s,
            times,
            tots,
            triggereds,
            0,      # event_id
        )
        self.blob = {'Hits': self.hits}

    def test_baseclass_selection_is_identity(self):
        hitsel = HitSelector()
        pre = self.blob['Hits'].serialise(to='pandas')
        hitsel.process(self.blob)
        post = self.blob['Hits'].serialise(to='pandas')
        assert_frame_equal(pre, post)
