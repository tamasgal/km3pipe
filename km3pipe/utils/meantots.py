#!/usr/bin/env python
# Filename: tots.py
"""
Convert ROOT and EVT files to HDF5.

Usage:
    meantots FILE [-p PLOT]
    meantots (-h | --help)

Options:
    -h --help   Show this screen.
    -p PLOT     Filename including extension [default: meantots.png].

"""

from collections import defaultdict

import km3pipe as kp
from km3pipe import version
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')    # noqa
import matplotlib.pyplot as plt
import km3pipe.style
from sklearn.mixture import GaussianMixture

from km3modules.common import StatusBar

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

km3pipe.style.use('km3pipe')


class MeanToTPlotter(kp.Module):
    """Create a plot of mean ToTs for each PMT."""

    def configure(self):
        self.tot_data = defaultdict(list)
        self.plotfilename = self.get("plotfilename") or "meantots.pdf"
        self.db = kp.db.DBManager()

    def process(self, blob):
        for hit in blob["TimesliceHits"]:
            self.tot_data[(hit.dom_id, hit.channel_id)].append(hit.tot)

    def finish(self):
        print("Calculating mean ToT for each PMT from gaussian fits...")
        gmm = GaussianMixture()
        xs, ys = [], []
        for (dom_id, channel_id), tots in self.tot_data.iteritems():
            dom = self.db.doms.via_dom_id(dom_id)
            gmm.fit(np.array(tots)[:, np.newaxis]).means_[0][0]
            mean_tot = gmm.means_[0][0]
            xs.append(31 * (dom.floor - 1) + channel_id + 600 * (dom.du - 1))
            ys.append(mean_tot)

        fig, ax = plt.subplots()
        ax.scatter(xs, ys, marker="+")
        ax.set_xlabel("31$\cdot$(floor - 1) + channel_id + 600$\cdot$(DU - 1)")
        ax.set_ylabel("ToT [ns]")
        plt.title("Mean ToT per PMT")
        plt.savefig(self.plotfilename)


class FastMeanToTPlotter(kp.Module):
    """Create a plot of mean ToTs for each PMT.

    This module is under development.
    """

    def configure(self):
        self.tot_data = defaultdict(list)
        self.plotfilename = self.get("plotfilename") or "meantots.pdf"
        self.db = kp.db.DBManager()

    def process(self, blob):
        hits = blob["TimesliceHits"]
        self.tot_data['dom_id'] += list(hits.dom_id)
        self.tot_data['channel_id'] += list(hits.channel_id)
        self.tot_data['tot'] += list(hits.tot)

    def finish(self):
        print("Calculating mean ToT for each PMT from gaussian fits...")
        gmm = GaussianMixture()
        xs, ys = [], []
        df = pd.DataFrame(self.tot_data)
        for (dom_id, channel_id), data in df.groupby(['dom_id', 'channel_id']):
            tots = data['tot']
            dom = self.db.doms.via_dom_id(dom_id)
            gmm.fit(tots[:, np.newaxis]).means_[0][0]
            mean_tot = gmm.means_[0][0]
            xs.append(31 * (dom.floor - 1) + channel_id + 600 * (dom.du - 1))
            ys.append(mean_tot)

        fig, ax = plt.subplots()
        ax.scatter(xs, ys, marker="+")
        ax.set_xlabel("31$\cdot$(floor - 1) + channel_id + 600$\cdot$(DU - 1)")
        ax.set_ylabel("ToT [ns]")
        plt.title("Mean ToT per PMT")
        plt.savefig(self.plotfilename)


def meantots(filename, plotfilename):
    pipe = kp.Pipeline()
    pipe.attach(kp.io.EventPump, filename=filename, with_timeslice_hits=True)
    pipe.attach(StatusBar, every=5000)
    pipe.attach(
        MeanToTPlotter, plotfilename=plotfilename, only_if="TimesliceHits"
    )
    pipe.drain(100000)


def main():
    from docopt import docopt
    args = docopt(__doc__, version=version)

    meantots(args["FILE"], args["-p"])
