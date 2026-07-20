# Filename: ztplot.py
"""
Create a zt-Plot.

Usage:
    ztplot [-t] -d DETX_FILE -e EVENT_ID FILE
    ztplot [-t] -d DETX_FILE -f FRAME -c COUNTER FILE
    ztplot (-h | --help)
    ztplot --version

Options:
    FILE          Input file.
    -c COUNTER    Trigger counter.
    -d DETX_FILE  Detector file.
    -e EVENT_ID   Event ID.
    -f FRAME      Frame index.
    -t            Triggered DUs only.
    -h, --help    Show this screen.

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


def find_event(pump, frame_index, trigger_counter):
    """Return the blob of the event with the given frame index and counter"""
    for blob in pump:
        event = blob["event"]
        if (
            event.frame_index == frame_index
            and event.trigger_counter == trigger_counter
        ):
            return blob
    return None


def plot_event(blob, detx_file, triggered_dus_only, title):
    """Create a zt-plot, one panel per DU"""
    cal = kp.calib.Calibration(filename=detx_file)
    hits = cal.apply(blob["event"].hits)
    det = cal.detector

    if triggered_dus_only:
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
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    plt.suptitle(title, fontsize=16)
    fig.text(0.5, 0.04, "time [ns]", ha="center")
    fig.text(0.08, 0.5, "z [m]", va="center", rotation="vertical")
    return fig


def main():
    from docopt import docopt

    arguments = docopt(__doc__, version=kp.version)

    filename = arguments["FILE"]
    pump = kp.io.OfflinePump(filename=filename)

    if arguments["-e"] is not None:
        event_id = int(arguments["-e"])
        blob = pump[event_id]
        subtitle = "Event #{0}".format(event_id)
    else:
        frame_index = int(arguments["-f"])
        trigger_counter = int(arguments["-c"])
        blob = find_event(pump, frame_index, trigger_counter)
        if blob is None:
            raise SystemExit(
                "No event with frame index {0} and trigger counter {1} in {2}.".format(
                    frame_index, trigger_counter, filename
                )
            )
        subtitle = "FrameIndex {0}, TriggerCounter {1}".format(
            frame_index, trigger_counter
        )

    plot_event(
        blob,
        arguments["-d"],
        arguments["-t"],
        "Filename: {0} - {1}".format(filename, subtitle),
    )
    plt.show()
