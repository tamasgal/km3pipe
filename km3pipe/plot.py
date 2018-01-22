# coding=utf-8
# Filename: plot.py
# pylint: disable=C0103
# pragma: no cover
"""
Common Plotting utils.
"""
from __future__ import division, absolute_import, print_function

import matplotlib as mpl    # noqa
import matplotlib.pyplot as plt
import seaborn as sns

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2018, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "BSD-3"
__maintainer__ = "Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"


def hexbin(x, y, color, **kwargs):
    """Seaborn-compatible hexbin plot.

    See also: http://seaborn.pydata.org/tutorial/axis_grids.html#mapping-custom-functions-onto-the-grid
    """
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, cmap=cmap,
               extent=[min(x), max(x), min(y), max(y)],
               **kwargs)
