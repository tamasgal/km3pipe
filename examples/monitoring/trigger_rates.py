#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: ts=4 sw=4 et
"""
====================
Trigger Rate Monitor
====================

A (messy) script to monitor the trigger rates.

"""
from __future__ import absolute_import, print_function, division

from datetime import datetime
from collections import defaultdict, deque, OrderedDict
import sys
from io import BytesIO
import os
import shutil
import time
import threading

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')    # noqa
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib.ticker as ticker

import km3pipe as kp
from km3pipe.config import Config
from km3pipe.io import CHPump
from km3pipe.io.daq import (DAQPreamble, DAQEvent)
import km3pipe.style
km3pipe.style.use('km3pipe')

log = kp.logger.get_logger("trigger_rate")

PLOTS_PATH = 'km3web/plots'

xfmt = md.DateFormatter('%Y-%m-%d %H:%M')
lock = threading.Lock()

general_style = dict(markersize=6, linestyle='None')
styles = {
    "Overall": dict(
        marker='D',
        markerfacecolor='None',
        markeredgecolor='tomato',
        markeredgewidth=1
    ),
    "3DMuon": dict(marker='X', markerfacecolor='dodgerblue'),
    "MXShower": dict(marker='v', markerfacecolor='orange'),
    "3DShower": dict(marker='o', markerfacecolor='greenyellow'),
}


def trigger_rate_sampling_period():
    try:
        return int(Config().get("Monitoring", "trigger_rate_sampling_period"))
    except (TypeError, ValueError):
        return 180


def is_3dshower(trigger_mask):
    return bool(trigger_mask & 1)


def is_mxshower(trigger_mask):
    return bool(trigger_mask & 2)


def is_3dmuon(trigger_mask):
    return bool(trigger_mask & 16)


class TriggerRate(kp.Module):
    def configure(self):
        self.run = True
        self.interval = self.get("interval") or trigger_rate_sampling_period()
        print("Update interval: {}s".format(self.interval))
        self.trigger_counts = defaultdict(int)
        self.trigger_rates = OrderedDict()
        for trigger in ["Overall", "3DMuon", "MXShower", "3DShower"]:
            self.trigger_rates[trigger] = deque(
                maxlen=int(60 * 24 / (self.interval / 60))
            )
        self.thread = threading.Thread(target=self.plot).start()

    def process(self, blob):
        if not str(blob['CHPrefix'].tag) == 'IO_EVT':
            return blob
        sys.stdout.write('.')
        sys.stdout.flush()

        data = blob['CHData']
        data_io = BytesIO(data)
        preamble = DAQPreamble(file_obj=data_io)    # noqa
        event = DAQEvent(file_obj=data_io)
        tm = event.trigger_mask
        with lock:
            self.trigger_counts["Overall"] += 1
            self.trigger_counts["3DShower"] += is_3dshower(tm)
            self.trigger_counts["MXShower"] += is_mxshower(tm)
            self.trigger_counts["3DMuon"] += is_3dmuon(tm)

        print(self.trigger_counts)

        return blob

    def plot(self):
        while self.run:
            time.sleep(self.interval)
            self.create_plot()

    def create_plot(self):
        print('\n' + self.__class__.__name__ + ": updating plot.")

        timestamp = datetime.utcnow()

        with lock:
            for trigger, n_events in self.trigger_counts.items():
                trigger_rate = n_events / self.interval
                self.trigger_rates[trigger].append((timestamp, trigger_rate))
            self.trigger_counts = defaultdict(int)

        fig, ax = plt.subplots(figsize=(16, 4))

        for trigger, rates in self.trigger_rates.items():
            timestamps, trigger_rates = zip(*rates)
            ax.plot(
                timestamps,
                trigger_rates,
                **styles[trigger],
                **general_style,
                label=trigger
            )
        ax.set_title(
            "Trigger Rates\n{0} UTC".format(datetime.utcnow().strftime("%c"))
        )
        ax.set_xlabel("time")
        ax.set_ylabel("trigger rate [Hz]")
        ax.xaxis.set_major_formatter(xfmt)
        ax.yaxis.set_major_locator(
            ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=100)
        )
        ax.grid(True)
        ax.minorticks_on()
        plt.legend()

        fig.tight_layout()

        filename = os.path.join(PLOTS_PATH, 'trigger_rates_lin_test.png')
        filename_tmp = os.path.join(
            PLOTS_PATH, 'trigger_rates_lin_test_tmp.png'
        )
        plt.savefig(filename_tmp, dpi=120, bbox_inches="tight")
        shutil.move(filename_tmp, filename)

        try:
            ax.set_yscale('log')
        except ValueError:
            pass

        filename = os.path.join(PLOTS_PATH, 'trigger_rates_test.png')
        filename_tmp = os.path.join(PLOTS_PATH, 'trigger_rates_test_tmp.png')
        plt.savefig(filename_tmp, dpi=120, bbox_inches="tight")
        shutil.move(filename_tmp, filename)

        plt.close('all')
        print("Plot updated at '{}'.".format(filename))

    def finish(self):
        self.run = False
        if self.thread is not None:
            self.thread.stop()


pipe = kp.Pipeline()
pipe.attach(
    CHPump,
    host='127.0.0.1',
    port=5553,
    tags='IO_EVT',
    timeout=60 * 60 * 24 * 7,
    max_queue=200000
)
pipe.attach(TriggerRate, interval=60)
pipe.drain()
