# Filename: test_plot.py
# pylint: disable=locally-disabled,C0111,R0904,C0103

import numpy as np

from km3pipe.testing import TestCase
from km3pipe.plot import bincenters

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"


class TestBins(TestCase):
    def test_binlims(self):
        bins = np.linspace(0, 20, 21)
        assert bincenters(bins).shape[0] == bins.shape[0] - 1
