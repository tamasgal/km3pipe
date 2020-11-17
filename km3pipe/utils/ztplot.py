# Filename: ztplot.py
"""
Create a zt-Plot.

Usage:
    ztplot [-t] -d DETX_FILE -e EVENT_ID FILE
    ztplot [-d DETX_FILE] [-t] -f FRAME -c COUNTER FILE
    ztplot (-h | --help)
    ztplot --version

Options:
    FILE          Input file.
    -c COUNTER    Trigger counter.
    -d DETX_FILE  Detector file.
    -e EVENT_ID   Event ID.
    -f FRAME      Frame index.
    -t            Triggered DUs only.
    -h --help     Show this screen.

"""

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np

import km3pipe as kp
import km3pipe.style  # noqa

km3pipe.style.use("km3pipe")

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def main():
    from docopt import docopt

    arguments = docopt(__doc__)

    try:
        event_id = int(arguments["-e"])
    except TypeError:
        pass
    else:
        pump = kp.io.OfflinePump(filename=arguments["FILE"])
        blob = pump[event_id]
        hits = blob["event"].hits
        cal = kp.calib.Calibration(filename=arguments["-d"])
        hits = cal.apply(hits)
        # triggered_dus = set(cal.detector.doms[h.dom_id][0] for h in hits)
        det = cal.detector
        if arguments["-t"]:
            dus = set(det.doms[h.dom_id][0] for h in hits if h.triggered)
        else:
            dus = set(det.doms[h.dom_id][0] for h in hits)

        n_plots = len(dus)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(n_plots / n_cols) + (n_plots % n_cols > 0)
        fig, axes = plt.subplots(
            ncols=n_cols, nrows=n_rows, sharex=True, sharey=True, figsize=(16, 16)
        )

        if n_cols == 1 and n_rows == 1:
            axes = (axes,)
        else:
            axes = axes.flatten()

        for ax, du in zip(axes, dus):
            du_hits = hits[hits.du == du]
            trig_hits = du_hits.triggered_rows

            ax.scatter(
                du_hits.time - min(du_hits.time),
                du_hits.pos_z,
                c="#09A9DE",
                label="hit",
            )
            ax.scatter(
                trig_hits.time - min(du_hits.time),
                trig_hits.pos_z,
                c="#FF6363",
                label="triggered hit",
            )
            ax.set_title("DU{0}".format(du), fontsize=8, fontweight="bold")

        for ax in axes:
            ax.tick_params(labelsize=8)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
            xlabels = ax.get_xticklabels()
            for label in xlabels:
                label.set_rotation(45)

        plt.suptitle(
            "Filename: {0} - Event #{1}".format(arguments["FILE"], event_id),
            fontsize=16,
        )
        fig.text(0.5, 0.04, "time [ns]", ha="center")
        fig.text(0.08, 0.5, "z [m]", va="center", rotation="vertical")
        #        plt.tight_layout()
        plt.show()
