# coding=utf-8
# Filename: plot.py
# pylint: disable=locally-disabled
"""
A collection of plotting functions and modules.

"""
from __future__ import division, absolute_import, print_function

from datetime import datetime
import os

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np

import km3pipe as kp
import km3pipe.style


def plot_dom_parameters(data, detector, filename, label, title,
                        vmin=0.0, vmax=10.0,
                        cmap='RdYlGn_r', under='deepskyblue', over='deeppink',
                        underfactor=1.0, overfactor=1.0,
                        missing='lightgray'):
    """Creates a plot in the classical monitoring.km3net.de style.

    Parameters
    ----------
    data: dict((du, floor) -> value)
    detector: km3pipe.hardware.Detector() instance
    filename: filename or filepath
    label: str
    title: str
    underfactor: a scale factor for the points used for underflow values
    overfactor: a scale factor for the points used for overflow values

    """
    x, y, _ = zip(*detector.doms.values())
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap(cmap)
    cmap.set_over(over, 1.0)
    cmap.set_under(under, 1.0)

    m_size = 100
    scatter_args = {
        'edgecolors': 'None',
        'vmin': vmin,
        'vmax': vmax,
    }
    sc_inactive = ax.scatter(x, y, c=missing, label='missing',
                             s=m_size*0.9,
                             **scatter_args)

    xa, ya = map(np.array, zip(*data.keys()))
    zs = np.array(list(data.values()))
    in_range_idx = np.logical_and(zs>=vmin, zs<=vmax)
    sc = ax.scatter(xa[in_range_idx], ya[in_range_idx],
                    c=zs[in_range_idx], cmap=cmap, s=m_size,
                    **scatter_args)
    under_idx = zs < vmin
    sc_under = ax.scatter(xa[under_idx], ya[under_idx],
                         c=under, label='< {0}'.format(vmin),
                         s=m_size*underfactor,
                         **scatter_args)
    over_idx = zs > vmax
    sc_over = ax.scatter(xa[over_idx], ya[over_idx],
                         c=over, label='> {0}'.format(vmax),
                         s=m_size*overfactor,
                         **scatter_args)

    cb = plt.colorbar(sc)
    cb.set_label(label)

    ax.set_title("{0}\n{1} UTC".format(title, datetime.utcnow().strftime("%c")))
    ax.set_xlabel("DU")
    ax.set_ylabel("DOM")
    ax.set_ylim(-2)
    ax.set_yticks(range(1, 18+1))
    major_locator = pylab.MaxNLocator(integer=True)
    sc_inactive.axes.xaxis.set_major_locator(major_locator)

    ax.legend(bbox_to_anchor=(0., -.16 , 1., .102), loc=1,
               ncol=2, mode="expand", borderaxespad=0.)

    fig.tight_layout()

    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close('all')
