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


def get_ax(ax=None):
    """Grab last ax if none specified, or pass through."""
    if ax is None:
        ax = plt.gca()
    return ax


def diag(ax=None, linecolor='0.0', linestyle='--', **kwargs):
    """Plot the diagonal."""
    ax = get_ax(ax)
    xy_min = np.min((ax.get_xlim(), ax.get_ylim()))
    xy_max = np.max((ax.get_ylim(), ax.get_xlim()))
    return ax.plot([xy_min, xy_max], [xy_min, xy_max],
                   ls=linestyle, c=linecolor, **kwargs)


def automeshgrid(x, y, step=0.02, xstep=None, ystep=None, pad=0.5,
                 xpad=None, ypad=None):
    """Make a meshgrid, inferred from data."""
    if xpad is None:
        xpad = pad
    if xstep is None:
        xstep = step
    if ypad is None:
        ypad = pad
    if ystep is None:
        ystep = step
    xmin = x.min() - xpad
    xmax = x.max() + xpad
    ymin = y.min() - ypad
    ymax = y.max() + ypad
    return meshgrid(xmin, xmax, step, ymin, ymax, ystep)


def meshgrid(x_min, x_max, x_step, y_min=None, y_max=None, y_step=None):
    """Make a meshgrid, e.g. when plotting decision surfaces."""
    if y_min is None:
        y_min = x_min
    if y_max is None:
        y_max = x_max
    if y_step is None:
        y_step = x_step
    xx, yy = np.meshgrid(np.arange(x_min, x_max, x_step),
                         np.arange(y_min, y_max, y_step))
    return xx, yy
