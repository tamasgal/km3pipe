#!usr/bin/env python
# Filename: stats.py
# pylint: disable=C0103
"""
Statistics.
"""
import numpy as np
from scipy.stats import rv_continuous

from .math import log_b
from .logger import logging

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2017, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"

log = logging.getLogger(__name__)  # pylint: disable=C0103


class loguniform(rv_continuous):
    """Loguniform Distributon"""
    def __init__(self, low=0.1, high=1, base=10, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self._log_low = log_b(low, base=base)
        self._log_high = log_b(high, base=base)
        self._base_of_log = base

    def _rvs(self, *args, **kwargs):
        # `rvs(size=foo, *args)` does argcheck etc, and sets `self._size`
        return np.power(self._base_of_log, np.random.uniform(
            self._log_low, self._log_high, self._size))


class rv_kde(rv_continuous):
    """Create a `scipy.stats.rv_continuous` instance from a (gaussian) KDE.

    Uses the KDE implementation from sklearn.

    Automatic bandwidth,  either from the statsmodels or scipy implementation.
    """
    def __init__(self, data, bw=None, bw_method=None, bw_statsmodels=False,
                 **kde_args):
        from sklearn.neighbors import KernelDensity
        from sklearn.utils import check_array
        data = check_array(data, order='C')
        if bw is None:
            if bw_statsmodels:
                bw = self._bandwidth_statsmodels(data, bw_method)
            else:
                bw = self._bandwidth_scipy(data, bw_method)
        self._bw = bw
        self._kde = KernelDensity(bandwidth=bw, **kde_args).fit(data)
        super(rv_kde, self).__init__(name='KDE')

    def _bandwidth_statsmodels(cls, sample, bw_method=None):
        from statsmodels.nonparametric.kernel_density import KDEMultivariate
        # all continuous
        vt = sample.ndim * 'c'
        skde = KDEMultivariate(sample, var_type=vt)
        bw = skde.bw
        return bw

    def _bandwidth_scipy(cls, sample, bw_method=None):
        from scipy.stats import gaussian_kde
        # sklearn expects switched shape versus scipy
        sample = sample.T
        gkde = gaussian_kde(sample, bw_method=None)
        f = gkde.covariance_factor()
        bw = f * sample.std()
        return bw

    def pdf(self, x):
        # we implement `pdf` instead of `_pdf`, since
        # otherwise scipy performs reshaping of `x` which messes
        # things up for sklearn -- we wanna reshape ourselves!
        x = check_array(x, order='C')
        log_pdf = self._kde.score_samples(x)
        pdf = np.exp(log_pdf)
        return pdf

    def _rvs(self, *args, random_state=None, **kwargs):
        # don't ask me why it uses `self._size`
        return np.exp(self._kde.sample(n_samples=self._size,
                                       random_state=random_state))


def mad(v):
    """MAD -- Median absolute deviation. More robust than standard deviation.
    """
    return np.median(np.abs(v - np.median(v)))


def drop_zero_variance(df):
    """Remove columns from dataframe with zero variance."""
    return df.copy().loc[:, df.var() != 0].copy()


def param_names(scipy_dist):
    """Get names of fit parameters from a ``scipy.rv_*`` distribution."""
    names = ['loc', 'scale']
    if scipy_dist.shapes is not None:
        names += scipy_dist.shapes.split()
    return names


def perc(p=95):
    """Create symmetric percentiles, with ``p`` coverage."""
    offset = (100 - p) / 2
    return offset, 100 - offset


def resample_1d(arr, n_out=None):
    """Resample an array, with replacement.

    Parameters
    ==========
    arr: np.ndarray
        The array is resampled along the first axis.
    n_out: int, optional
        Number of samples to return. If not specified,
        return ``len(arr)`` samples.
    """
    arr = np.atleast_1d(arr)
    n = len(arr)
    if n_out is None:
        n_out = n
    idx = np.random.randint(0, n, size=n)
    return arr[idx]


def bootstrap_params(rv_cont, data, n_iter=5):
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
        params = rv_cont.fit(resample_1d(data))
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


def bootstrap_fit(rv_cont, data, n_iter=10, quant=95, print_params=True):
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
        print("{nam:>{fill}}: {mean:+.3f} ∈ [{lo:+.3f}, {up:+.3f}] ({q}%)".format(
            nam=name, fill=maxlen, mean=par[i], lo=lo[i], up=up[i], q=quant))
    out = {
        'mean': par,
        'lower limit': lo,
        'upper limit': up,
    }
    return out
