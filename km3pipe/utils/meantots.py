#!/usr/bin/env python
# coding=utf-8
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
from __future__ import division, absolute_import, print_function

from collections import defaultdict
import sys

import km3pipe as kp
from km3pipe import version
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import km3pipe.style
from sklearn.mixture import GaussianMixture

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def meantots(filename, plotfilename):
    db = kp.db.DBManager()
    pump = kp.io.JPPPump(filename=filename)

    tot_data = defaultdict(list)
    for b in pump:
        for hit in b["Hits"]:
            tot_data[(hit.dom_id, hit.channel_id)].append(hit.tot)

    gmm = GaussianMixture()
    xs, ys = [], []
    for (dom_id, channel_id), tots in tot_data.iteritems():
        dom = db.doms.via_dom_id(dom_id)
        gmm.fit(np.array(tots)[:, np.newaxis]).means_[0][0]
        mean_tot = gmm.means_[0][0]
        xs.append(31*(dom.floor - 1) + channel_id + 600*(dom.du-1))
        ys.append(mean_tot)

    fig, ax = plt.subplots()
    ax.scatter(xs, ys, marker="+")
    ax.set_xlabel("31$\cdot$(floor - 1) + channel_id + 600$\cdot$(DU - 1)")
    ax.set_ylabel("ToT [ns]")
    plt.title("Mean ToT per PMT\n{0}".format(filename))
    plt.savefig(plotfilename)


def main():
    from docopt import docopt
    args = docopt(__doc__, version=version)

    meantots(args["FILE"], args["PLOT"])
