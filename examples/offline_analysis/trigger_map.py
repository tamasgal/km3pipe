#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: ts=4 sw=4 et
"""
===========
Trigger Map
===========

This script creates a histogram to show the trigger contribution for events.

.. code-block:: bash

    Usage:
        trigger_map.py [-d DU -p PLOT_FILENAME] FILENAME
        trigger_map.py --version

    Option:
        FILENAME          Name of the input file.
        -d DU             Only plot for the given DU.
        -p PLOT_FILENAME  The filename of the plot [default: trigger_map.png].
        -h --help         Show this screen.

"""
from __future__ import absolute_import, print_function, division

from docopt import docopt
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')    # noqa
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import km3pipe as kp
import km3pipe.style


class TriggerMap(kp.Module):
    """Creates a plot to show the number of triggered hits for each DOM."""

    def configure(self):
        self.det = self.require('detector')
        self.plot_filename = self.require('plot_filename')
        self.subtitle = self.get('subtitle', default='')
        self.du = self.get('du')
        if self.du is not None:
            self.n_dus = 1
            self.n_doms = 18
        else:
            self.n_dus = self.det.n_dus
            self.n_doms = int(self.det.n_doms / self.n_dus)
        self.hit_counts = []

    def process(self, blob):
        hits = blob['Hits'].triggered_rows
        dom_ids = np.unique(hits.dom_id)
        hit_counts = np.zeros(self.n_dus * self.n_doms)
        for dom_id in dom_ids:
            n_hits = np.sum(hits.dom_id == dom_id)
            du, floor, _ = self.det.doms[dom_id]
            if self.du is not None and du != self.du:
                continue
            if self.du:
                du = 1
            hit_counts[(du - 1) * self.n_doms + floor - 1] += n_hits
        self.hit_counts.append(hit_counts)

        return blob

    def finish(self):
        self.create_plot()

    def create_plot(self):
        km3pipe.style.use("km3pipe")

        title = "Trigger Map\n{}".format(self.subtitle)
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.grid(True)
        ax.set_axisbelow(True)
        hit_mat = np.array([np.array(x) for x in self.hit_counts]).transpose()
        im = ax.matshow(
            hit_mat,
            interpolation='nearest',
            filternorm=None,
            cmap='plasma',
            aspect='auto',
            origin='lower',
            zorder=3,
            norm=LogNorm(vmin=1, vmax=np.amax(hit_mat))
        )
        yticks = np.arange(self.n_doms * self.n_dus)
        ytick_label_templ = "DU{0:.0f}-DOM{1:02d}" if self.du else "DOM{1:02d}"
        ytick_labels = [
            ytick_label_templ.format(
                np.ceil((y + 1) / self.n_doms), y % (self.n_doms) + 1
            ) for y in yticks
        ]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        ax.tick_params(labelbottom=True)
        ax.tick_params(labeltop=True)
        ax.set_xlabel("event id")
        ax.set_title(title)
        cb = fig.colorbar(im, pad=0.05)
        cb.set_label("number of triggered hits")

        fig.tight_layout()

        plt.savefig(self.plot_filename, dpi=120, bbox_inches="tight")


if __name__ == '__main__':
    args = docopt(__doc__, version='1.0')
    print(args)
    du = int(args['-d']) if args['-d'] else None
    det = kp.hardware.Detector(det_id=29)
    pipe = kp.Pipeline()
    pipe.attach(kp.io.jpp.EventPump, filename=args['FILENAME'])
    pipe.attach(
        TriggerMap,
        detector=det,
        du=du,
        plot_filename=args['-p'],
        subtitle=args['FILENAME']
    )
    pipe.drain()
