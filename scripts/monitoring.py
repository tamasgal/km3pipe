#!/usr/bin/env python
# vim: ts=4 sw=4 et

__author__ = "Tamas Gal"

from datetime import datetime
from collections import deque, defaultdict
from functools import partial
from io import StringIO
import os
import shutil
import time
import threading

import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib.colors import LogNorm
from matplotlib import pylab
import pandas as pd
import numpy as np

from km3pipe import Pipeline, Module
from km3pipe.hardware import Detector
from km3pipe.io import CHPump
from km3pipe.io.daq import DAQPreamble, DAQSummaryslice, DAQEvent
from km3pipe.time import tai_timestamp
import km3pipe.style

km3pipe.style.use('km3pipe')

PLOTS_PATH = '/home/km3net/monitoring/www/plots'
N_DOMS = 18
N_DUS = 2
detector = Detector(det_id=14)

xfmt = md.DateFormatter('%Y-%m-%d %H:%M')
lock = threading.Lock()


class DOMHits(Module):
    def configure(self):
        self.run = True
        self.max_events = 1000
        self.hits = deque(maxlen=1000)
        self.triggered_hits = deque(maxlen=1000)
        self.thread = threading.Thread(target=self.plot).start()

    def process(self, blob):
        tag = str(blob['CHPrefix'].tag)

        if not tag == 'IO_EVT':
            return blob

        data = blob['CHData']
        data_io = StringIO(data)
        preamble = DAQPreamble(file_obj=data_io)    # noqa
        event = DAQEvent(file_obj=data_io)
        with lock:
            hits = np.zeros(N_DOMS * N_DUS)
            for dom_id, _, _, _ in event.snapshot_hits:
                du, floor, _ = detector.doms[dom_id]
                hits[(du - 1) * N_DOMS + floor - 1] += 1
            self.hits.append(hits)
            triggered_hits = np.zeros(N_DOMS * N_DUS)
            for dom_id, _, _, _, _ in event.triggered_hits:
                du, floor, _ = detector.doms[dom_id]
                triggered_hits[(du - 1) * N_DOMS + floor - 1] += 1
            self.triggered_hits.append(triggered_hits)

        return blob

    def plot(self):
        while self.run:
            with lock:
                self.create_plots()
            time.sleep(10)

    def create_plots(self):
        if len(self.hits) > 0:
            self.create_plot(self.hits, "Hits on DOMs", 'hits_on_doms')
        if len(self.triggered_hits) > 0:
            self.create_plot(
                self.triggered_hits, "Triggered Hits on DOMs",
                'triggered_hits_on_doms'
            )

    def create_plot(self, hits, title, filename):
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.grid(True)
        ax.set_axisbelow(True)
        hit_matrix = np.array([np.array(x) for x in hits]).transpose()
        im = ax.matshow(
            hit_matrix,
            interpolation='nearest',
            filternorm=None,
            cmap='plasma',
            aspect='auto',
            origin='lower',
            zorder=3,
            norm=LogNorm(vmin=1, vmax=np.amax(hit_matrix))
        )
        yticks = np.arange(N_DOMS * N_DUS)
        ytick_labels = [
            "DU{0:0.0f}-DOM{1:02d}".format(
                np.ceil((y + 1) / N_DOMS), y % (N_DOMS) + 1
            ) for y in yticks
        ]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        ax.tick_params(labelbottom=False)
        ax.tick_params(labeltop=False)
        ax.set_xlabel("event (latest on the right)")
        ax.set_title(
            "{0} - via the last {1} Events\n{2}".format(
                title, self.max_events, time.strftime("%c")
            )
        )
        cb = fig.colorbar(im, pad=0.05)
        cb.set_label("number of hits")

        fig.tight_layout()

        f = os.path.join(PLOTS_PATH, filename + '.png')
        f_tmp = os.path.join(PLOTS_PATH, filename + '_tmp.png')
        plt.savefig(f_tmp, dpi=120, bbox_inches="tight")
        plt.close('all')
        shutil.move(f_tmp, f)

    def finish(self):
        self.run = False
        if self.thread is not None:
            self.thread.stop()


class TriggerRate(Module):
    def configure(self):
        self.run = True
        self.event_times = deque(maxlen=4000)
        self.trigger_rates = deque(maxlen=4000)
        self.thread = threading.Thread(target=self.plot).start()
        self.store = pd.HDFStore('data/trigger_rates.h5')
        self.restore_data()

    def restore_data(self):
        with lock:
            data = zip(
                self.store.trigger_rates.timestamp,
                self.store.trigger_rates.rate
            )
            self.trigger_rates.extend(data)
            print("{0} data points restored.".format(len(self.trigger_rates)))

    def process(self, blob):
        tag = str(blob['CHPrefix'].tag)

        if not tag == 'IO_EVT':
            return blob

        data = blob['CHData']
        data_io = StringIO(data)
        preamble = DAQPreamble(file_obj=data_io)    # noqa
        event = DAQEvent(file_obj=data_io)
        timestamp = event.header.time_stamp
        with lock:
            self.event_times.append(timestamp)

        return blob

    def plot(self):
        while self.run:
            print("Obtaining lock")
            with lock:
                self.create_plot()
            print("Releasing lock")
            time.sleep(10)

    def create_plot(self):
        print(self.__class__.__name__ + ": updating plot.")
        timestamp = time.time()
        now = datetime.fromtimestamp(timestamp)
        interval = 60
        n_events = sum(t > timestamp - interval for t in self.event_times)
        rate = n_events / 60
        self.trigger_rates.append((now, rate))
        self.store.append(
            'trigger_rates', pd.DataFrame({
                'timestamp': [now],
                'rate': [rate]
            })
        )
        print(
            "Number of rates recorded: {0} (last: {1}".format(
                len(self.trigger_rates), self.trigger_rates[-1]
            )
        )

        x, y = zip(*self.trigger_rates)
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.xaxis.set_major_formatter(xfmt)
        data = pd.DataFrame({'dates': x, 'rates': y})
        data.plot('dates', 'rates', grid=True, ax=ax, legend=False, style='.')
        ax.set_title(
            "Trigger Rate - via Event Times\n{0}".format(time.strftime("%c"))
        )
        ax.set_xlabel("time")
        ax.set_ylabel("trigger rate [Hz]")
        try:
            ax.set_yscale('log')
        except ValueError:
            pass

        fig.tight_layout()

        filename = os.path.join(PLOTS_PATH, 'trigger_rates.png')
        filename_tmp = os.path.join(PLOTS_PATH, 'trigger_rates_tmp.png')
        plt.savefig(filename_tmp, dpi=120, bbox_inches="tight")
        plt.close('all')
        shutil.move(filename_tmp, filename)

    def finish(self):
        self.run = False
        self.store.close()
        if self.thread is not None:
            self.thread.stop()


class DOMActivityPlotter(Module):
    def configure(self):
        self.index = 0
        self.rates = defaultdict(partial(deque, maxlen=4000))
        self.run = True
        self.thread = threading.Thread(target=self.plot, args=()).start()

    def process(self, blob):
        self.index += 1
        if self.index % 30:
            return blob

        tag = str(blob['CHPrefix'].tag)
        data = blob['CHData']

        if not tag == 'IO_SUM':
            return blob

        data = blob['CHData']
        data_io = StringIO(data)
        preamble = DAQPreamble(file_obj=data_io)    # noqa
        summaryslice = DAQSummaryslice(file_obj=data_io)
        timestamp = summaryslice.header.time_stamp
        with lock:
            for dom_id, rates in summaryslice.summary_frames.iteritems():
                du, dom, _ = detector.doms[dom_id]
                self.rates[(du, dom)].append((timestamp, sum(rates)))

        return blob

    def plot(self):
        while self.run:
            with lock:
                self.create_plot()
            time.sleep(5)

    def create_plot(self):
        x, y, _ = zip(*detector.doms.values())
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap('RdYlGn_r')
        cmap.set_over('deeppink', 1.0)
        cmap.set_under('deepskyblue', 1.0)

        vmax = 15 * 60

        scatter_args = {
            'edgecolors': 'None',
            's': 100,
            'vmin': 0.0,
            'vmax': vmax,
        }
        sc_inactive = ax.scatter(
            x, y, c='lightgray', label='inactive', **scatter_args
        )
        now = tai_timestamp()

        try:
            xa, ya = map(np.array, zip(*self.rates.keys()))
            ts = np.array([now - max(zip(*d)[0]) for d in self.rates.values()])
        except ValueError:
            print("Not enough data.")
            pass
        else:
            active_idx = ts < vmax
            sc_active = ax.scatter(
                xa[active_idx],
                ya[active_idx],
                c=ts[active_idx],
                cmap=cmap,
                **scatter_args
            )
            ax.scatter(
                xa[~active_idx],
                ya[~active_idx],
                c='deeppink',
                label='> {0} s'.format(vmax),
                **scatter_args
            )
            cb = plt.colorbar(sc_active)
            cb.set_label("last activity [s]")

        ax.set_title(
            "DOM Activity - via Summary Slices\n{0}".format(
                time.strftime("%c")
            )
        )
        ax.set_xlabel("DU")
        ax.set_ylabel("DOM")
        ax.set_ylim(-2)
        ax.set_yticks(range(1, N_DOMS + 1))
        major_locator = pylab.MaxNLocator(integer=True)
        sc_inactive.axes.xaxis.set_major_locator(major_locator)

        ax.legend(
            bbox_to_anchor=(0., -.16, 1., .102),
            loc=1,
            ncol=2,
            mode="expand",
            borderaxespad=0.
        )

        fig.tight_layout()

        filename = os.path.join(PLOTS_PATH, 'dom_activity.png')
        filename_tmp = os.path.join(PLOTS_PATH, 'dom_activity_tmp.png')
        plt.savefig(filename_tmp, dpi=120, bbox_inches="tight")
        plt.close('all')
        shutil.move(filename_tmp, filename)

    def finish(self):
        self.run = False
        if self.thread is not None:
            self.thread.stop()


pipe = Pipeline()
pipe.attach(
    CHPump,
    host='192.168.0.110',
    port=5553,
    tags='IO_EVT, IO_SUM',
    timeout=60 * 60 * 24 * 7,
    max_queue=2000
)
pipe.attach(DOMActivityPlotter)
pipe.attach(TriggerRate)
pipe.attach(DOMHits)
pipe.drain()
