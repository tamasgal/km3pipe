#!usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: stats.py
# pylint: disable=C0103
"""
Statistics.
"""
from __future__ import absolute_import, print_function, division

import numpy as np
from scipy.stats import rv_continuous

from .math import log_b
from .logger import get_logger
from .tools import zero_pad

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2017, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"

log = get_logger(__name__)    # pylint: disable=C0103


class loguniform(rv_continuous):
    """Loguniform Distributon"""

    def __init__(self, low=0.1, high=1, base=10, *args, **kwargs):
        super(loguniform, self).__init__(*args, **kwargs)
        self._log_low = log_b(low, base=base)
        self._log_high = log_b(high, base=base)
        self._base_of_log = base

    def _rvs(self, *args, **kwargs):
        # `rvs(size=foo, *args)` does argcheck etc, and sets `self._size`
        return np.power(
            self._base_of_log,
            np.random.uniform(self._log_low, self._log_high, self._size)
        )


class rv_kde(rv_continuous):
    """Create a `scipy.stats.rv_continuous` instance from a (gaussian) KDE.

    Uses the KDE implementation from sklearn.

    Automatic bandwidth,  either from the statsmodels or scipy implementation.
    """

    def __init__(
            self, data, bw=None, bw_method=None, bw_statsmodels=False,
            **kde_args
    ):
        from sklearn.neighbors import KernelDensity
        from sklearn.utils import check_array
        data = check_array(data, order='C')
        if bw is None:
            if bw_statsmodels:
                bw = self._bandwidth_statsmodels(data)
            else:
                bw = self._bandwidth_scipy(data, bw_method)
        self._kde = KernelDensity(bandwidth=bw, **kde_args).fit(data)
        self.bandwidth = bw
        self.kernel = self._kde.get_params()['kernel']
        super(rv_kde, self).__init__(name='KDE')

    @staticmethod
    def _bandwidth_statsmodels(sample):
        from statsmodels.nonparametric.kernel_density import KDEMultivariate
        # all continuous
        vt = sample.shape[1] * 'c'
        skde = KDEMultivariate(sample, var_type=vt)
        bw = skde.bw
        return bw

    @staticmethod
    def _bandwidth_scipy(sample, bw_method=None):
        from scipy.stats import gaussian_kde
        # sklearn expects switched shape versus scipy
        sample = sample.T
        gkde = gaussian_kde(sample, bw_method=bw_method)
        f = gkde.covariance_factor()
        bw = f * sample.std()
        return bw

    def pdf(self, x):
        # we implement `pdf` instead of `_pdf`, since
        # otherwise scipy performs reshaping of `x` which messes
        # things up for sklearn -- we wanna reshape ourselves!
        from sklearn.utils import check_array
        x = check_array(x, order='C')
        log_pdf = self._kde.score_samples(x)
        pdf = np.exp(log_pdf)
        return pdf

    def rvs(self, *args, **kwargs):
        """Draw Random Variates.

        Parameters
        ----------
        size: int, optional (default=1)
        random_state_: optional (default=None)
        """
        # TODO REVERSE THIS FUCK PYTHON2
        size = kwargs.pop('size', 1)
        random_state = kwargs.pop('size', None)
        # don't ask me why it uses `self._size`
        return self._kde.sample(n_samples=size, random_state=random_state)


def mad(v):
    """MAD -- Median absolute deviation. More robust than standard deviation.
    """
    return np.median(np.abs(v - np.median(v)))


def mad_std(v):
    """Robust estimate of standard deviation using the MAD.

    Lifted from astropy.stats."""
    MAD = mad(v)
    return MAD * 1.482602218505602


def drop_zero_variance(df):
    """Remove columns from dataframe with zero variance."""
    return df.copy().loc[:, df.var() != 0].copy()


def param_names(scipy_dist):
    """Get names of fit parameters from a ``scipy.rv_*`` distribution."""
    if not isinstance(scipy_dist, rv_continuous):
        raise TypeError
    names = ['loc', 'scale']
    if scipy_dist.shapes is not None:
        names += scipy_dist.shapes.split()
    return names


def perc(arr, p=95, **kwargs):
    """Create symmetric percentiles, with ``p`` coverage."""
    offset = (100 - p) / 2
    return np.percentile(arr, (offset, 100 - offset), **kwargs)


def resample_1d(arr, n_out=None, random_state=None):
    """Resample an array, with replacement.

    Parameters
    ==========
    arr: np.ndarray
        The array is resampled along the first axis.
    n_out: int, optional
        Number of samples to return. If not specified,
        return ``len(arr)`` samples.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    arr = np.atleast_1d(arr)
    n = len(arr)
    if n_out is None:
        n_out = n
    idx = random_state.randint(0, n, size=n)
    return arr[idx]


def bootstrap_params(rv_cont, data, n_iter=5, **kwargs):
    """Bootstrap the fit params of a distribution.

    Parameters
    ==========
    rv_cont: scipy.stats.rv_continuous instance
        The distribution which to fit.
    data: array-like, 1d
        The data on which to fit.
    n_iter: int [default=10]
        Number of bootstrap iterations.
    """
    fit_res = []
    for i in range(n_iter):
        params = rv_cont.fit(resample_1d(data, **kwargs))
        fit_res.append(params)
    fit_res = np.array(fit_res)
    return fit_res


def param_describe(params, quant=95, axis=0):
    """Get mean + quantile range from bootstrapped params."""
    par = np.mean(params, axis=axis)
    lo, up = perc(quant)
    p_up = np.percentile(params, up, axis=axis)
    p_lo = np.percentile(params, lo, axis=axis)
    return par, p_lo, p_up


def bootstrap_fit(
        rv_cont, data, n_iter=10, quant=95, print_params=True, **kwargs
):
    """Bootstrap a distribution fit + get confidence intervals for the params.

    Parameters
    ==========
    rv_cont: scipy.stats.rv_continuous instance
        The distribution which to fit.
    data: array-like, 1d
        The data on which to fit.
    n_iter: int [default=10]
        Number of bootstrap iterations.
    quant: int [default=95]
        percentile of the confidence limits (default is 95, i.e. 2.5%-97.5%)
    print_params: bool [default=True]
        Print a fit summary.
    """
    fit_params = bootstrap_params(rv_cont, data, n_iter)
    par, lo, up = param_describe(fit_params, quant=quant)
    names = param_names(rv_cont)
    maxlen = max([len(s) for s in names])
    print("--------------")
    print(rv_cont.name)
    print("--------------")
    for i, name in enumerate(names):
        print(
            "{nam:>{fill}}: {mean:+.3f} ∈ "
            "[{lo:+.3f}, {up:+.3f}] ({q}%)".format(
                nam=name,
                fill=maxlen,
                mean=par[i],
                lo=lo[i],
                up=up[i],
                q=quant
            )
        )
    out = {
        'mean': par,
        'lower limit': lo,
        'upper limit': up,
    }
    return out


class hist2d(rv_continuous):
    """Simple implementation of a 2d histogram."""

    def __init__(self, H2D, *args, **kwargs):
        H, xlims, ylims = H2D
        self.H = H
        self.xlims = xlims
        self.ylims = ylims
        self.H /= self.integral
        self.xcenters = bincenters(xlims)
        self.ycenters = bincenters(ylims)
        super(hist2d, self).__init__(*args, **kwargs)

    @property
    def H_pad(self):
        return zero_pad(self.H)

    def _pdf(self, X):
        X = np.atleast_2d(X)
        x = X[:, 0]
        y = X[:, 1]
        return self.H_pad[np.searchsorted(self.xlims, x, side='right'),
                          np.searchsorted(self.ylims, y, side='right'),
                          ]

    @property
    def integral(self):
        return (self.H * self.area).sum()

    @property
    def area(self):
        xwidths = np.diff(self.xlims)
        ywidths = np.diff(self.ylims)
        area = np.outer(xwidths, ywidths)
        return area


def bincenters(bins):
    """Bincenters, assuming they are all equally spaced."""
    bins = np.atleast_1d(bins)
    return 0.5 * (bins[1:] + bins[:-1])
