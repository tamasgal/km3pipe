# Filename: plot.py
# -*- coding: utf-8 -*-
# pylint: disable=locally-disabled
"""
A collection of plotting functions and modules.

"""
from __future__ import absolute_import, print_function, division

from datetime import datetime
import os
import multiprocessing as mp
import time

import pandas as pd
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt    # noqa
from matplotlib import pylab    # noqa

try:
    import healpy as hp
except ImportError:
    pass

import km3pipe as kp    # noqa
import km3pipe.style    # noqa


def plot_dom_parameters(
        data,
        detector,
        filename,
        label,
        title,
        vmin=0.0,
        vmax=10.0,
        cmap='RdYlGn_r',
        under='deepskyblue',
        over='deeppink',
        underfactor=1.0,
        overfactor=1.0,
        missing='lightgray',
        hide_limits=False
):
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
    hide_limits: do not show under/overflows in the plot

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
    sc_inactive = ax.scatter(
        x, y, c=missing, label='missing', s=m_size * 0.9, **scatter_args
    )

    xa, ya = map(np.array, zip(*data.keys()))
    zs = np.array(list(data.values()))
    in_range_idx = np.logical_and(zs >= vmin, zs <= vmax)
    sc = ax.scatter(
        xa[in_range_idx],
        ya[in_range_idx],
        c=zs[in_range_idx],
        cmap=cmap,
        s=m_size,
        **scatter_args
    )
    if not hide_limits:
        under_idx = zs < vmin
        ax.scatter(
            xa[under_idx],
            ya[under_idx],
            c=under,
            label='< {0}'.format(vmin),
            s=m_size * underfactor,
            **scatter_args
        )
        over_idx = zs > vmax
        ax.scatter(
            xa[over_idx],
            ya[over_idx],
            c=over,
            label='> {0}'.format(vmax),
            s=m_size * overfactor,
            **scatter_args
        )

    cb = plt.colorbar(sc)
    cb.set_label(label)

    ax.set_title(
        "{0}\n{1} UTC".format(title,
                              datetime.utcnow().strftime("%c"))
    )
    ax.set_xlabel("DU")
    ax.set_ylabel("DOM")
    ax.set_ylim(-2)
    ax.set_yticks(range(1, 18 + 1))
    major_locator = pylab.MaxNLocator(integer=True)
    sc_inactive.axes.xaxis.set_major_locator(major_locator)

    ax.legend(
        bbox_to_anchor=(0., -.16, 1., .102),
        loc=1,
        ncol=2,
        mode="expand",
        borderaxespad=0.
    )

    fig.tight_layout()

    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close('all')


def make_dom_map(pmt_directions, values, nside=512, d=0.2, smoothing=0.1):
    """Create a mollweide projection of a DOM with given PMTs.

    The output can be used to call the `healpy.mollview` function.
    """
    discs = [hp.query_disc(nside, dir, 0.2) for dir in pmt_directions]
    npix = hp.nside2npix(nside)
    pixels = np.zeros(npix)
    for disc, value in zip(discs, values):
        for d in disc:
            pixels[d] = value
    if smoothing > 0:
        return hp.sphtfunc.smoothing(pixels, fwhm=smoothing, iter=1)
    return pixels


class IntraDOMCalibrationPlotter(kp.Module):
    def configure(self):
        self.plots_path = self.get('plots_path', default=os.getcwd())
        self.data_path = self.get('data_path', default=os.getcwd())
        self.det_oid = self.require('det_oid')
        self.db = kp.db.DBManager()

    def process(self, blob):
        calibration = blob["IntraDOMCalibration"]
        for process in (self.create_plot, self.save_hdf5):
            proc = mp.Process(target=process, args=(calibration, ))
            proc.daemon = True
            proc.start()
            proc.join()
        return blob

    def create_plot(self, calibration):
        print("Creating plot...")
        fig, axes = plt.subplots(
            6, 3, figsize=(16, 20), sharex=True, sharey=True
        )
        for ax, (dom_id, calib) in zip(axes.flatten(), calibration.items()):
            ax.plot(np.cos(calib['angles']), calib["means"], '.')
            ax.plot(np.cos(calib['angles']), calib["corrected_means"], '.')
            ax.set_title(
                "{0} - {1}".format(
                    self.db.doms.via_dom_id(dom_id, self.det_oid), dom_id
                )
            )
            ax.set_ylim((-10, 10))
        plt.suptitle("{0} UTC".format(datetime.utcnow().strftime("%c")))
        plt.savefig(
            os.path.join(self.plots_path, "intradom.png"), bbox_inches='tight'
        )
        plt.close('all')

        fig, axes = plt.subplots(
            6, 3, figsize=(16, 20), sharex=True, sharey=True
        )
        for ax, (dom_id, calib) in zip(axes.flatten(), calibration.items()):
            ax.plot(np.cos(calib['angles']), calib["rates"], '.')
            ax.plot(np.cos(calib['angles']), calib["corrected_rates"], '.')
            ax.set_title(
                "{0} - {1}".format(
                    self.db.doms.via_dom_id(dom_id, self.det_oid), dom_id
                )
            )
            ax.set_ylim((0, 10))
        plt.suptitle("{0} UTC".format(datetime.utcnow().strftime("%c")))
        plt.savefig(
            os.path.join(self.plots_path, "angular_k40rate_distribution.png"),
            bbox_inches='tight'
        )
        plt.close('all')

    def save_hdf5(self, calibration):
        print("Saving calibration information...")
        store = pd.HDFStore(
            os.path.join(self.data_path, 'k40calib.h5'), mode='a'
        )
        now = int(time.time())
        timestamps = (now, ) * 31
        for dom_id, calib in calibration.items():
            tdc_channels = range(31)
            t0s = calib['opt_t0s'].x
            dom_ids = (dom_id, ) * 31
            df = pd.DataFrame({
                'timestamp': timestamps,
                'dom_id': dom_ids,
                'tdc_channel': tdc_channels,
                't0s': t0s
            })
            store.append('t0s', df, format='table', data_columns=True)
        store.close()
