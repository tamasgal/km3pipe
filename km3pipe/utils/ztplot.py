# coding=utf-8
# Filename: ztplot.py
"""
Create a zt-Plot.

Usage:
    ztplot [-d DETX_FILE] [-t] -e EVENT_ID FILE
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
from __future__ import division, absolute_import, print_function

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np

import km3pipe as kp
from km3pipe.dataclasses import HitSeries
import km3pipe.style  # noqa

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
        event_id = int(arguments['-e'])
    except TypeError:
        pass
    else:
        pump = kp.GenericPump(arguments['FILE'])
        blob = pump[event_id]
        event_info = blob['EventInfo']
        hits = blob['Hits']
        if(arguments['-d']):
            geo = kp.core.Geometry(filename=arguments['-d'])
        else:
            geo = kp.core.Geometry(det_id=event_info.det_id)
        geo.apply(hits)
        # triggered_dus = set(geo.detector.doms[h.dom_id][0] for h in hits)
        det = geo.detector
        if arguments['-t']:
            dus = set(det.doms[h.dom_id][0] for h in hits if h.triggered)
        else:
            dus = set(det.doms[h.dom_id][0] for h in hits)

        n_plots = len(dus)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(n_plots / n_cols) + (n_plots % n_cols > 0)
        fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows,
                                 sharex=True, sharey=True, figsize=(16, 16))

        # [axes.flat[-1 - i].axis('off') for i in range(len(axes.flat) - n_plots)]

        for ax, du in zip(axes.flat, dus):
            _hits = [h for h in hits
                     if det.doms[h.dom_id][0] == du]
            du_hits = HitSeries(_hits)
            trig_hits = HitSeries([h for h in _hits if h.triggered])

            ax.scatter(du_hits.time, [z for (x, y, z) in du_hits.pos],
                       c='#09A9DE', label='hit')
            ax.scatter(trig_hits.time, [z for (x, y, z) in trig_hits.pos],
                       c='#FF6363', label='triggered hit')
            ax.set_title('DU{0}'.format(du), fontsize=8, fontweight='bold')

        for ax in axes.flat:
            ax.tick_params(labelsize=8)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
            xlabels = ax.get_xticklabels()
            for label in xlabels:
                label.set_rotation(45)

        plt.suptitle("Filename: {0} - Event #{1}"
                     .format(arguments['FILE'], event_id), fontsize=16)
        fig.text(0.5, 0.04, 'time [ns]', ha='center')
        fig.text(0.08, 0.5, 'z [m]', va='center', rotation='vertical')
#        plt.tight_layout()
        plt.show()
