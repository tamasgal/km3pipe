# Filename: plot.py
# pylint: disable=C0103
# pragma: no cover
"""
Common Plotting utils.
"""
try:
    import _tkinter  # noqa
except ImportError:
    import matplotlib

    matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from km3pipe.stats import bincenters
import km3pipe.extras

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2018, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "BSD-3"
__maintainer__ = "Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"


def hexbin(x, y, color="purple", **kwargs):
    """Seaborn-compatible hexbin plot.

    See also: http://seaborn.pydata.org/tutorial/axis_grids.html#mapping-custom-functions-onto-the-grid
    """
    if HAS_SEABORN:
        cmap = sns.light_palette(color, as_cmap=True)
    else:
        cmap = "Purples"
    plt.hexbin(x, y, cmap=cmap, **kwargs)


def get_ax(ax=None):
    """Grab last ax if none specified, or pass through."""
    if ax is None:
        ax = plt.gca()
    return ax


def diag(ax=None, linecolor="0.0", linestyle="--", **kwargs):
    """Plot the diagonal."""
    ax = get_ax(ax)
    xy_min = np.min((ax.get_xlim(), ax.get_ylim()))
    xy_max = np.max((ax.get_ylim(), ax.get_xlim()))
    return ax.plot(
        [xy_min, xy_max], [xy_min, xy_max], ls=linestyle, c=linecolor, **kwargs
    )


def automeshgrid(
    x, y, step=0.02, xstep=None, ystep=None, pad=0.5, xpad=None, ypad=None
):
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
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, x_step), np.arange(y_min, y_max, y_step)
    )
    return xx, yy


def prebinned_hist(counts, binlims, ax=None, *args, **kwargs):
    """Plot a histogram with counts, binlims already given.

    Examples
    ========
    >>> gaus = np.random.normal(size=100)
    >>> counts, binlims = np.histogram(gaus, bins='auto')
    >>> prebinned_hist(countsl binlims)
    """
    ax = get_ax(ax)
    x = bincenters(binlims)
    weights = counts
    return ax.hist(x, bins=binlims, weights=weights, *args, **kwargs)


def joint_hex(x, y, **kwargs):
    """Seaborn Joint Hexplot with marginal KDE + hists."""
    return sns.jointplot(
        x, y, kind="hex", stat_func=None, marginal_kws={"kde": True}, **kwargs
    )


def plot_convexhull(xy, ax=None, plot_points=True):
    scipy = km3pipe.extras.scipy()
    from scipy.spatial import ConvexHull

    ch = ConvexHull(xy)
    ax = get_ax(ax)
    if plot_points:
        ax.plot(xy[:, 0], xy[:, 1], "o")
    for simplex in ch.simplices:
        ax.plot(xy[simplex, 0], xy[simplex, 1], "k-")
    return ax
