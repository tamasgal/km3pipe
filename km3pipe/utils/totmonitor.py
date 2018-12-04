#!/usr/bin/env python
# Filename: totmonitor.py
"""
Display the mean ToT for each TDC channel for given DOM.

Usage:
    totmonitor [options] DOM_ID
    totmonitor (-h | --help)
    totmonitor --version

Options:
    -l LIGIER       Ligier address [default: 127.0.0.1].
    -p PORT         Ligier port [default: 5553].
    -o OPTIMAL_TOT  Target ToT in ns [default: 26.4].
    -t TOLERANCE    Defines the range for valid ToT [default: 0.3].
    -u UPDATE       Update frequency in seconds [default: 10].
    -h --help       Show this screen.
"""

import os
from collections import defaultdict

from km3pipe.dataclasses import Table
from km3pipe.dev import cprint
from km3pipe.core import Module

import numpy as np
from sklearn import mixture

__author__ = "Tamas Gal and Jonas Reubelt"
__copyright__ = "Copyright 2017, the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Jonas Reubelt"
__email__ = "tgal@km3net.de"
__status__ = "Development"

GMM = mixture.GaussianMixture(n_components=1)


class TimesliceCreator(Module):
    """Create `TimesliceHitSeries` from raw timeslice hits."""

    def configure(self):
        self.dom_id = self.require("dom_id")

    def process(self, blob):
        hits = blob['TimesliceFrames'][self.dom_id]
        n_hits = len(hits)
        if n_hits == 0:
            return blob
        channel_ids, times, tots = zip(*hits)
        ts_hits = Table({
            'channel_id': np.array(channel_ids),
            'dom_id': np.full(n_hits, self.dom_id),
            'time': np.array(times),
            'tot': np.array(tots),
            'group_id': 0
        })
        blob['TimesliceHits'] = ts_hits
        blob['DOM_ID'] = self.dom_id
        return blob


class MeanTotDisplay(Module):
    def configure(self):
        self.optimal_tot = self.get("optimal_tot") or 26.4
        self.tolerance = self.get("tolerance") or 0.3
        self.update_frequency = self.get("update_frequency") or 10
        self.tots = defaultdict(list)
        self.counter = 0
        self.dom_id = None

    def process(self, blob):
        hits = blob["TimesliceHits"]
        self.dom_id = blob["DOM_ID"]
        for channel in range(31):
            idx = hits.channel_id == channel
            tots = hits.tot[idx]
            self.tots[channel] += list(tots)

        self.counter += 1
        if self.counter % int(self.update_frequency * 10) == 0:
            self.update_display()
            self.tots = defaultdict(list)
            self.counter = 0
        return blob

    def update_display(self):
        os.system('clear')
        self.print_header()
        for channel, tots in self.tots.items():
            if channel % 8 == 0:
                self.print_scale()
            GMM.fit(np.array(tots)[:, np.newaxis])
            mean_tot = GMM.means_[np.argmin(GMM.covariances_)][0]
            # mean_tot = np.median(tots)
            if np.isnan(mean_tot):
                mean_tot = 0
            color = 'green'
            if mean_tot > self.optimal_tot + self.tolerance:
                color = 'red'
            if mean_tot < self.optimal_tot - self.tolerance:
                color = 'blue'
            cprint(
                "Channel {0:02d}: {1:.1f}ns    {2}".format(
                    channel, mean_tot,
                    int(mean_tot) * '|'
                ), color
            )
        self.print_scale()
        self.print_footer()

    def print_header(self):
        print(
            "Mean ToT (average over {0}s) for DOM: {1}".format(
                self.update_frequency, self.dom_id
            )
        )
        print(
            "                     "
            "0         10        20        30        40        50"
        )

    def print_footer(self):
        print(
            "                     "
            "0         10        20        30        40        50"
        )

    def print_scale(self):
        print("                     " + '|----+----' * 5)


def main():
    from docopt import docopt
    import km3pipe as kp
    args = docopt(__doc__)
    pipe = kp.Pipeline()
    pipe.attach(
        kp.io.ch.CHPump,
        host=args['-l'],
        port=int(args['-p']),
        tags='IO_TSL',
        max_queue=100,
        timeout=60 * 60 * 24
    )
    pipe.attach(kp.io.daq.TimesliceParser)
    pipe.attach(TimesliceCreator, dom_id=int(args['DOM_ID']))
    pipe.attach(
        MeanTotDisplay,
        only_if="TimesliceHits",
        optimal_tot=float(args['-o']),
        update_frequency=float(args['-u']),
        tolerance=float(args['-t'])
    )
    pipe.drain()


if __name__ == "__main__":
    main()
