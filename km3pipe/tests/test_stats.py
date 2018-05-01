# Filename: test_stats.py
# pylint: disable=locally-disabled,C0111,R0904,C0103

import numpy as np

from km3pipe.testing import TestCase
from km3pipe.stats import loguniform

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
        assert r.shape == (100,)
        assert np.all(r <= hi)
        assert np.all(r >= lo)
        dist = loguniform(low=lo, high=hi, base=2)
        r2 = dist.rvs(size=500)
        assert r2.shape == (500,)
        assert np.all(r2 <= hi)
        assert np.all(r2 >= lo)
