#!usr/bin/env python
# Filename: stats.py
# pylint: disable=C0103
"""
Statistics.
"""
import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_array
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from .logger import logging

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2017, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"

log = logging.getLogger(__name__)  # pylint: disable=C0103


class loguniform(stats.rv_continuous):
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


class rv_kde(stats.rv_continuous):
    """Create a `scipy.stats.rv_continuous` instance from a (gaussian) KDE.

    Uses the KDE implementation from sklearn.

    Automatic bandwidth,  either from the statsmodels or scipy implementation.
    """
    def __init__(self, data, bw=None, bw_method=None, bw_statsmodels=False,
                 **kde_args):
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
        # all continuous
        vt = sample.ndim * 'c'
        skde = KDEMultivariate(sample, var_type=vt)
        bw = skde.bw
        return bw

    def _bandwidth_scipy(cls, sample, bw_method=None):
        # sklearn expects switched shape versus scipy
        sample = sample.T
        gkde = stats.gaussian_kde(sample, bw_method=None)
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
