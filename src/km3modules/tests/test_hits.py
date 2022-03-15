# Filename: test_hits.py
# -*- coding: utf-8 -*-
# vim:set ts=4 sts=4 sw=4 et:
"""
Tests for Hit functions and Modules.
"""

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from km3modules.hits import count_multiplicities

from km3pipe import Table, Blob, Pipeline, Module
from km3pipe.testing import TestCase


class TestMultiplicityCounter(TestCase):
    def test_count_multiplicities(self):
        times = np.array([1, 10, 20, 22, 30, 34, 40, 50])
        mtps, coinc_ids = count_multiplicities(times, tmax=20)
        self.assertListEqual([3, 3, 3, 4, 4, 4, 4, 1], list(mtps))
        self.assertListEqual([0, 0, 0, 1, 1, 1, 1, 2], list(coinc_ids))

    def test_count_other_multiplicities(self):
        times = np.array([1, 10, 20, 22, 30, 34, 40, 50])
        mtps, coinc_ids = count_multiplicities(times, tmax=10)
        self.assertListEqual([2, 2, 3, 3, 3, 2, 2, 1], list(mtps))
        self.assertListEqual([0, 0, 1, 1, 1, 2, 2, 3], list(coinc_ids))
