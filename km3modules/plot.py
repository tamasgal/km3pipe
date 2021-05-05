# Filename: plot.py
# -*- coding: utf-8 -*-
# pylint: disable=locally-disabled
"""
A collection of plotting functions and modules.

"""
from collections import Counter
from datetime import datetime
import os
import multiprocessing as mp
import time

import numpy as np
import matplotlib.pyplot as plt  # noqa
import matplotlib.ticker as ticker
from matplotlib import pylab  # noqa
import km3db

import km3pipe as kp  # noqa
import km3pipe.style  # noqa

from km3modules.hits import count_multiplicities


def plot_dom_parameters(
    data,
    detector,
    filename,
    label,
    title,
    vmin=0.0,
    vmax=10.0,
    cmap="cividis",
    under="deepskyblue",
    over="deeppink",
    underfactor=1.0,
    overfactor=1.0,
    missing="lightgray",
    hide_limits=False,
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
        "edgecolors": "None",
        "vmin": vmin,
        "vmax": vmax,
    }
    sc_inactive = ax.scatter(
        x, y, c=missing, label="missing", s=m_size * 0.9, **scatter_args
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
            label="< {0}".format(vmin),
            s=m_size * underfactor,
            **scatter_args
        )
        over_idx = zs > vmax
        ax.scatter(
            xa[over_idx],
            ya[over_idx],
            c=over,
            label="> {0}".format(vmax),
            s=m_size * overfactor,
            **scatter_args
        )

    cb = plt.colorbar(sc)
    cb.set_label(label)

    ax.set_title("{0}\n{1} UTC".format(title, datetime.utcnow().strftime("%c")))
    ax.set_xlabel("DU")
    ax.set_ylabel("DOM")
    ax.set_ylim(-2)
    ax.set_yticks(range(1, 18 + 1))
    major_locator = pylab.MaxNLocator(integer=True)
    sc_inactive.axes.xaxis.set_major_locator(major_locator)

    ax.legend(
        bbox_to_anchor=(0.0, -0.16, 1.0, 0.102),
        loc=1,
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
    )

    fig.tight_layout()

    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close("all")


def make_dom_map(pmt_directions, values, nside=512, d=0.2, smoothing=0.1):
    """Create a mollweide projection of a DOM with given PMTs.

    The output can be used to call the `healpy.mollview` function.
    """
    import healpy as hp

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
        self.plots_path = self.get("plots_path", default=os.getcwd())
        self.data_path = self.get("data_path", default=os.getcwd())
        self.det_oid = self.require("det_oid")
        self.clbmap = km3db.CLBMap(self.det_oid)

    def process(self, blob):
        calibration = blob["IntraDOMCalibration"]
        for process in (self.create_plot, self.save_hdf5):
            proc = mp.Process(target=process, args=(calibration,))
            proc.daemon = True
            proc.start()
            proc.join()
        return blob

    def create_plot(self, calibration):
        print("Creating plot...")
        fig, axes = plt.subplots(6, 3, figsize=(16, 20), sharex=True, sharey=True)
        sorted_dom_ids = sorted(
            calibration.keys(),
            key=lambda d: (self.clbmap.dom_ids[d].du, self.clbmap.dom_ids[d].floor),
        )  # by DU and FLOOR, note that DET OID is needed!
        for ax, dom_id in zip(axes.flatten(), sorted_dom_ids):
            calib = calibration[dom_id]
            ax.plot(np.cos(calib["angles"]), calib["means"], ".")
            ax.plot(np.cos(calib["angles"]), calib["corrected_means"], ".")
            du = self.clbmap.dom_ids[dom_id].du
            floor = self.clbmap.dom_ids[dom_id].floor
            ax.set_title("{0} - {1}".format("DU{}-DOM{}".format(du, floor), dom_id))
            ax.set_ylim((-10, 10))
        plt.suptitle("{0} UTC".format(datetime.utcnow().strftime("%c")))
        plt.savefig(os.path.join(self.plots_path, "intradom.png"), bbox_inches="tight")
        plt.close("all")

        fig, axes = plt.subplots(6, 3, figsize=(16, 20), sharex=True, sharey=True)
        for ax, dom_id in zip(axes.flatten(), sorted_dom_ids):
            calib = calibration[dom_id]
            ax.plot(np.cos(calib["angles"]), calib["rates"], ".")
            ax.plot(np.cos(calib["angles"]), calib["corrected_rates"], ".")
            du = self.clbmap.dom_ids[dom_id].du
            floor = self.clbmap.dom_ids[dom_id].floor
            ax.set_title("{0} - {1}".format("DU{}-DOM{}".format(du, floor), dom_id))
            ax.set_ylim((0, 10))
        plt.suptitle("{0} UTC".format(datetime.utcnow().strftime("%c")))
        plt.savefig(
            os.path.join(self.plots_path, "angular_k40rate_distribution.png"),
            bbox_inches="tight",
        )
        plt.close("all")

    def save_hdf5(self, calibration):
        print("Saving calibration information...")
        pd = kp.extras.pandas()
        store = pd.HDFStore(os.path.join(self.data_path, "k40calib.h5"), mode="a")
        now = int(time.time())
        timestamps = (now,) * 31
        for dom_id, calib in calibration.items():
            tdc_channels = range(31)
            t0s = calib["opt_t0s"].x
            dom_ids = (dom_id,) * 31
            df = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "dom_id": dom_ids,
                    "tdc_channel": tdc_channels,
                    "t0s": t0s,
                }
            )
            store.append("t0s", df, format="table", data_columns=True)
        store.close()


def ztplot(
    hits,
    filename=None,
    title=None,
    max_z=None,
    figsize=(16, 8),
    n_dus=4,
    ytick_distance=200,
    max_multiplicity_entries=10,
    grid_lines=[],
):
    """Creates a ztplot like shown in the online monitoring"""
    fontsize = 16

    dus = set(hits.du)

    if n_dus is not None:
        dus = [c[0] for c in Counter(hits.du).most_common(n_dus)]
        mask = [du in dus for du in hits.du]
        hits = hits[mask]

    dus = sorted(dus)
    doms = set(hits.dom_id)

    hits = hits.append_columns("multiplicity", np.ones(len(hits))).sorted(by="time")

    if max_z is None:
        max_z = int(np.ceil(np.max(hits.pos_z) / 100.0)) * 100 * 1.05

    for dom in doms:
        dom_hits = hits[hits.dom_id == dom]
        mltps, m_ids = count_multiplicities(dom_hits.time)
        hits["multiplicity"][hits.dom_id == dom] = mltps

    if np.any(hits.triggered):
        time_offset = np.min(hits[hits.triggered > 0].time)
        hits.time -= time_offset

    n_plots = len(dus)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(n_plots / n_cols) + (n_plots % n_cols > 0)
    marker_fig, marker_axes = plt.subplots()
    # for the marker size hack...
    fig, axes = plt.subplots(
        ncols=n_cols,
        nrows=n_rows,
        sharex=True,
        sharey=True,
        figsize=figsize,
        constrained_layout=True,
    )

    axes = [axes] if n_plots == 1 else trim_axes(axes, n_plots)

    for ax, du in zip(axes, dus):
        du_hits = hits[hits.du == du]
        for grid_line in grid_lines:
            ax.axhline(grid_line, lw=1, color="b", ls="--", alpha=0.15)
        trig_hits = du_hits[du_hits.triggered > 0]

        ax.scatter(
            du_hits.time,
            du_hits.pos_z,
            s=du_hits.multiplicity * 30,
            c="#09A9DE",
            label="hit",
            alpha=0.5,
        )
        ax.scatter(
            trig_hits.time,
            trig_hits.pos_z,
            s=trig_hits.multiplicity * 30,
            alpha=0.8,
            marker="+",
            c="#FF6363",
            label="triggered hit",
        )
        ax.set_title("DU{0}".format(int(du)), fontsize=fontsize, fontweight="bold")

        # The only way I could create a legend with matching marker sizes
        max_multiplicity = int(np.max(du_hits.multiplicity))
        markers = list(
            range(
                0,
                max_multiplicity,
                np.ceil(max_multiplicity / max_multiplicity_entries).astype(int),
            )
        )
        custom_markers = [
            marker_axes.scatter([], [], s=mult * 30, color="#09A9DE", lw=0, alpha=0.5)
            for mult in markers
        ] + [marker_axes.scatter([], [], s=30, marker="+", c="#FF6363")]
        ax.legend(
            custom_markers,
            ["multiplicity"] + ["       %d" % m for m in markers[1:]] + ["triggered"],
            scatterpoints=1,
            markerscale=1,
            loc="lower right",
            frameon=True,
            framealpha=0.7,
        )

    for idx, ax in enumerate(axes):
        ax.set_ylim(0, max_z)
        ax.tick_params(labelsize=fontsize)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(ytick_distance))
        xlabels = ax.get_xticklabels()
        for label in xlabels:
            label.set_rotation(45)

        if idx % n_cols == 0:
            ax.set_ylabel("z [m]", fontsize=fontsize)
        if idx >= len(axes) - n_cols:
            ax.set_xlabel("time [ns]", fontsize=fontsize)

    if title is not None:
        plt.suptitle(title, fontsize=fontsize, y=1.05)
    if filename is not None:
        plt.savefig(filename, dpi=120, bbox_inches="tight")

    return fig


def trim_axes(axes, n):
    """little helper to massage the axes list to have correct length..."""
    axes = axes.flat
    for ax in axes[n:]:
        ax.remove()
    return axes[:n]


def cumulative_run_livetime(qtable, kind="runs"):
    """Create a figure which plots the cumulative livetime of runs

    Parameters
    ----------
    qtable: pandas.DataFrame
        A table which has the run number as index and columns for
        'livetime_s', 'timestamp' and 'datetime' (pandas datetime).
    kind: str
        'runs' to plot for each run or 'timeline' to plot based
        on the actual run time.

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots()

    options = {
        "runs": {"xlabel": "run", "xs": qtable.index},
        "timeline": {"xlabel": None, "xs": qtable.datetime},
    }

    actual_livetime = np.cumsum(qtable["livetime_s"])
    optimal_livetime = np.cumsum(qtable.timestamp.diff())

    ax.plot(options[kind]["xs"], actual_livetime, label="actual livetime")
    ax.plot(options[kind]["xs"], optimal_livetime, label="100% livetime")
    ax.set_xlabel(options[kind]["xlabel"])
    ax.set_ylabel("time / s")
    ax.legend()

    return fig
