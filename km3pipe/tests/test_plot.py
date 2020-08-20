# Filename: test_plot.py
# pylint: disable=locally-disabled,C0111,R0904,C0103

import numpy as np

from km3pipe.testing import TestCase, patch
from km3pipe.plot import meshgrid, automeshgrid, diag

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"


class TestMeshStuff(TestCase):
    def test_meshgrid(self):
        xx, yy = meshgrid(-1, 1, 0.8)
        assert np.allclose(
            [[-1.0, -0.2, 0.6], [-1.0, -0.2, 0.6], [-1.0, -0.2, 0.6]], xx
        )
        assert np.allclose(
            [[-1.0, -1.0, -1.0], [-0.2, -0.2, -0.2], [0.6, 0.6, 0.6]], yy
        )

    def test_meshgrid_with_y_specs(self):
        xx, yy = meshgrid(-1, 1, 0.8, -10, 10, 8)
        assert np.allclose(
            [[-1.0, -0.2, 0.6], [-1.0, -0.2, 0.6], [-1.0, -0.2, 0.6]], xx
        )
        assert np.allclose([[-10, -10, -10], [-2, -2, -2], [6, 6, 6]], yy)
