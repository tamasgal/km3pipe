# Filename: test_stats.py
# pylint: disable=locally-disabled,C0111,R0904,C0103

import numpy as np
from numpy.random import RandomState
from pandas import DataFrame
import pytest

from km3pipe.testing import TestCase
from km3pipe.stats import mad, mad_std, drop_zero_variance, perc, bincenters

__author__ = ["Tamas Gal", "Moritz Lotze"]
__copyright__ = "Copyright 2016, KM3Pipe devs and the KM3NeT collaboration."
__credits__ = ["Thomas Heid"]
__license__ = "MIT"
__maintainer__ = ["Tamas Gal", "Moritz Lotze"]
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestMAD(TestCase):
    def test_wiki(self):
        arr = np.array([1, 1, 2, 2, 4, 6, 9])
        assert np.allclose(mad(arr), 1)

    def test_normal(self):
        np.random.seed(42)
        sample = np.random.normal(0, 1, 1000)
        assert np.allclose(mad_std(sample), 1, atol=0.1)


class TestPerc(TestCase):
    def test_settings(self):
        arr = np.array(
            [
                -2.394,
                0.293,
                0.371,
                0.384,
                1.246,
            ]
        )
        assert np.allclose(perc(arr), [-2.1253, 1.1598])
        assert np.allclose(perc(arr, interpolation="nearest"), [arr[0], arr[-1]])


class TestVariance(TestCase):
    def test_easy(self):
        df = DataFrame({"a": [1, 2, 3], "b": [1, 1, 1], "c": [4, 4, 3]})
        df2 = drop_zero_variance(df)
        assert df.shape == (3, 3)
        assert df2.shape == (3, 2)
        assert "b" not in df2.columns
        assert "b" in df.columns


class TestBins(TestCase):
    def test_binlims(self):
        bins = np.linspace(0, 20, 21)
        assert bincenters(bins).shape[0] == bins.shape[0] - 1
