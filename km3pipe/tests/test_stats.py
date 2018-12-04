# Filename: test_stats.py
# pylint: disable=locally-disabled,C0111,R0904,C0103

import numpy as np
from numpy.random import RandomState
from pandas import DataFrame
import pytest
from scipy import stats

from km3pipe.testing import TestCase
from km3pipe.stats import (
    loguniform, rv_kde, mad, mad_std, drop_zero_variance, param_names, perc,
    resample_1d, bootstrap_params, param_describe, bootstrap_fit, hist2d,
    bincenters
)

__author__ = ["Tamas Gal", "Moritz Lotze"]
__copyright__ = "Copyright 2016, KM3Pipe devs and the KM3NeT collaboration."
__credits__ = ["Thomas Heid"]
__license__ = "MIT"
__maintainer__ = ["Tamas Gal", "Moritz Lotze"]
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestLogUniform(TestCase):
    def setUp(self):
        np.random.seed(1234)

    def test_rvs(self):
        lo, hi = 0.1, 10
        dist = loguniform(low=lo, high=hi, base=10)
        r = dist.rvs(size=100)
        assert r.shape == (100, )
        assert np.all(r <= hi)
        assert np.all(r >= lo)
        dist = loguniform(low=lo, high=hi, base=2)
        r2 = dist.rvs(size=500)
        assert r2.shape == (500, )
        assert np.all(r2 <= hi)
        assert np.all(r2 >= lo)


class TestRvKde(TestCase):
    def setUp(self):
        self.data1d = np.random.normal(size=20)
        self.data = self.data1d[:, np.newaxis]

    def test_shape(self):
        with pytest.raises(ValueError):
            rv = rv_kde(self.data1d)
        rv = rv_kde(self.data)
        samp = rv.rvs(size=10)
        assert len(samp) == 10
        assert samp.ndim == 2
        assert samp.shape == (10, 1)
        samp = rv.rvs(size=(10, ))
        assert len(samp) == 10
        assert samp.ndim == 2
        assert samp.shape == (10, 1)

    def test_2d(self):
        d2d = np.column_stack([
            np.random.normal(size=20),
            np.random.normal(size=20),
        ])
        rv = rv_kde(d2d)
        samp = rv.rvs(size=10)
        assert len(samp) == 10
        assert samp.ndim == 2
        assert samp.shape == (10, 2)

    def test_bw_methods(self):
        rv = rv_kde(
            self.data,
            bw_method='scott',
        )
        assert rv is not None

    def test_bw_statsmodels(self):
        rv = rv_kde(
            self.data,
            bw_statsmodels=True,
        )
        assert rv is not None

    def test_bw_statsmodels_ignores_method(self):
        rv1 = rv_kde(
            self.data,
            bw_statsmodels=True,
        )
        rv2 = rv_kde(
            self.data,
            bw_statsmodels=True,
            bw_method='scott',
        )
        rv3 = rv_kde(
            self.data,
            bw_statsmodels=True,
            bw_method='silverman',
        )
        assert rv2 is not None
        assert rv1 is not None
        assert rv1.bandwidth == rv2.bandwidth
        assert rv1.bandwidth == rv3.bandwidth
        assert rv2.bandwidth == rv3.bandwidth


class TestMAD(TestCase):
    def test_wiki(self):
        arr = np.array([1, 1, 2, 2, 4, 6, 9])
        assert np.allclose(mad(arr), 1)

    def test_normal(self):
        np.random.seed(42)
        sample = np.random.normal(0, 1, 1000)
        assert np.allclose(mad_std(sample), 1, atol=.1)


class TestPerc(TestCase):
    def test_settings(self):
        arr = np.array([
            -2.394,
            0.293,
            0.371,
            0.384,
            1.246,
        ])
        assert np.allclose(perc(arr), [-2.1253, 1.1598])
        assert np.allclose(
            perc(arr, interpolation='nearest'), [arr[0], arr[-1]]
        )


class TestVariance(TestCase):
    def test_easy(self):
        df = DataFrame({'a': [1, 2, 3], 'b': [1, 1, 1], 'c': [4, 4, 3]})
        df2 = drop_zero_variance(df)
        assert df.shape == (3, 3)
        assert df2.shape == (3, 2)
        assert 'b' not in df2.columns
        assert 'b' in df.columns


class TestResample(TestCase):
    def test_shape(self):
        arr = np.array([1, 1, 2, 2, 4, 6, 9])
        r = resample_1d(arr)
        assert arr.shape == r.shape

    def test_seed(self):
        seed = 42
        arr = np.array([1, 1, 2, 2, 4, 6, 9])
        r1 = resample_1d(arr, random_state=RandomState(seed=seed))
        r2 = resample_1d(arr, random_state=RandomState(seed=seed))
        assert np.allclose(r1, r2)
        assert np.allclose([9, 2, 4, 9, 2, 4, 4], r1)


class TestBootstrapFit(TestCase):
    def setUp(self):
        self.seed = 42

    def test_raise_on_instance(self):
        with pytest.raises(TypeError):
            n = param_names(stats.norm())
            assert n is not None

    def test_names(self):
        assert param_names(stats.norm) == ['loc', 'scale']
        assert param_names(stats.lognorm) == ['loc', 'scale', 's']

    def test_raw(self):
        arr = np.array([1, 1, 2, 2, 4, 6, 9])
        pars = bootstrap_params(
            stats.norm, arr, n_iter=500, random_state=RandomState(self.seed)
        )
        assert np.allclose(
            np.mean(pars, axis=0), [np.mean(arr), np.std(arr)], atol=.5
        )

    def test_param_describe(self):
        arr = np.array([1, 1, 2, 2, 4, 6, 9])
        pars = bootstrap_params(
            stats.norm, arr, n_iter=500, random_state=RandomState(self.seed)
        )
        desc = param_describe(pars)
        assert desc is not None
        assert len(desc) == 3
        assert desc[0].shape[0] == 2

    def test_full(self):
        arr = np.array([1, 1, 2, 2, 4, 6, 9])
        fits = bootstrap_fit(
            stats.norm, arr, n_iter=500, random_state=RandomState(self.seed)
        )
        assert np.allclose(fits['mean'], [np.mean(arr), np.std(arr)], atol=.5)


class TestHist2D(TestCase):
    def setUp(self):
        np.random.seed(42)
        self.sample = np.random.normal(0, 1, (100, 2))
        self.bins = ([-2, 0, 2], [-2, 0, 2])

    def test_pdf(self):
        hist = np.histogram2d(
            self.sample[:, 0], self.sample[:, 1], normed=True, bins=self.bins
        )
        h = hist2d(hist)
        assert h.H.shape == (2, 2)
        assert h.H_pad.shape == (4, 4)
        f = h.pdf([[0, 0]])
        assert f is not None

    def test_norm(self):
        hist1 = np.histogram2d(
            self.sample[:, 0], self.sample[:, 1], normed=True, bins=self.bins
        )
        hist2 = np.histogram2d(
            self.sample[:, 0], self.sample[:, 1], normed=False, bins=self.bins
        )
        h1 = hist2d(hist1)
        h2 = hist2d(hist2)

        print(h1.area)
        print(h1.H)
        print(h1.H / h1.area)
        print((h1.H / h1.area).sum())
        print(h1.H.sum())

        h1s = h1.H.sum()
        assert np.allclose(h1s, h1.H.sum())

        assert np.allclose(h1.H.sum(), h2.H.sum())

        assert np.allclose(h1.integral, 1)
        assert np.allclose(h2.integral, 1)


class TestBins(TestCase):
    def test_binlims(self):
        bins = np.linspace(0, 20, 21)
        assert bincenters(bins).shape[0] == bins.shape[0] - 1
