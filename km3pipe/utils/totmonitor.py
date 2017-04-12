#!/usr/bin/env python
# coding=utf-8
# Filename: totmonitor.py
"""
Display the mean ToT for each TDC channel for given DOM.

Usage:
    totmonitor [-l LIGIER] [-p PORT] [-o OPTIMAL_TOT] [-t tolerance] DOM_ID
    totmonitor (-h | --help)
    totmonitor --version

Options:
    -l LIGIER       Ligier address [default: 127.0.0.1].
    -p PORT         Ligier port [default: 5553].
    -o OPTIMAL_TOT  Target ToT in ns [default: 26.4].
    -t TOLERANCE    Defines the range for valid ToT [default: 0.3].
    -h --help       Show this screen.
"""
from __future__ import division

import os
from collections import defaultdict

from termcolor import colored, cprint

import km3pipe as kp
import numpy as np

__author__ = "Tamas Gal and Jonas Reubelt"
__copyright__ = "Copyright 2017, the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Jonas Reubelt"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TimesliceCreator(kp.core.Module):
    """Create `TimesliceHitSeries` from raw timeslice hits."""
    def configure(self):
        self.dom_id = self.require("dom_id")

    def process(self, blob):
        hits =  blob['TimesliceFrames'][self.dom_id]
        n_hits = len(hits)
        channel_ids, times, tots = zip(*hits)
        ts_hits = kp.dataclasses.TimesliceHitSeries.from_arrays(
                  np.array(channel_ids),
                  np.full(n_hits, self.dom_id),
                  np.array(times),
                  np.array(tots),
                  0,
                  0)
        blob['TimesliceHits'] = ts_hits
        return blob


class MeanTotDisplay(kp.core.Module):
    def configure(self):
        self.tots = defaultdict(list)
        self.counter = 0

    def process(self, blob):
        hits = blob["TimesliceHits"]
        for channel in range(31):
            idx = hits.channel_id == channel
            tots = hits.tot[idx]
            self.tots[channel] += list(tots)

        self.counter += 1
        if self.counter % 100 == 0:
            self.update_display()
            self.tots = defaultdict(list)
            self.counter = 0
        return blob

    def update_display(self):
        os.system('clear')
        self.print_header()
        for channel, tots in self.tots.iteritems():
            if channel % 8 == 0:
                self.print_scale()
            mean_tot = np.mean(tots)
            if np.isnan(mean_tot):
                mean_tot = 0
            color = 'green'
            if mean_tot > self.optimal_tot + self.tolreance:
                color = 'red'
            if mean_tot < self.optimal_tot - self.tolreance:
                color = 'blue'
            cprint("Channel {0:02d}: {1:.1f}ns    {2}"
                   .format(channel, mean_tot, int(mean_tot) * '|'),
                   color)
        self.print_scale()
        self.print_header()

    def print_header(self):
        print("                     "
              "0         10        20        30        40        50")

    def print_scale(self):
        print("                     " + '|----+----' * 10)


def main():
    from docopt import docopt
    args = docopt(__doc__)
    pipe = kp.Pipeline()
    pipe.attach(kp.io.ch.CHPump,
                host=args['-l'],
                port=int(args['-p']),
                tags='IO_TSL',
                max_queue=100,
                timeout=60*60*24)
    pipe.attach(kp.io.daq.TimesliceParser)
    pipe.attach(TimesliceCreator, dom_id=int(args['DOM_ID']))
    pipe.attach(MeanTotDisplay,
                optimal_tot=float(args['-o']),
                tolreance=float(args['-t']))
    pipe.drain()


if __name__ == "__main__":
    main()
