#!/usr/bin/env python
# vim: ts=4 sw=4 et

from datetime import datetime
from collections import deque, defaultdict
from functools import partial
from io import StringIO
from queue import Queue, Empty
import os
import shutil
import time
import threading

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as md
from matplotlib.colors import LogNorm
from matplotlib import pylab
import pandas as pd
import numpy as np

from km3pipe import Pipeline, Module
from km3pipe.calib import Calibration
from km3pipe.dataclasses import HitSeries
from km3pipe.io import CHPump
from km3pipe.io.daq import (
    DAQProcessor, DAQPreamble, DAQSummaryslice, DAQEvent
)
from km3pipe.time import tai_timestamp
import km3pipe.style

km3pipe.style.use('km3pipe')

PLOTS_PATH = '/home/km3net/monitoring/www/plots'
N_DOMS = 18
N_DUS = 2
cal = Calibration(det_id=14)
detector = cal.detector

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
            time.sleep(50)

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
                title, self.max_events,
                datetime.utcnow().strftime("%c")
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
        self.interval = 60
        self.event_times = deque(maxlen=4000)    # max events per interval
        self.trigger_rates = deque(maxlen=60 * 48)    # minutes
        self.thread = threading.Thread(target=self.plot).start()
        self.store = pd.HDFStore('data/trigger_rates.h5', 'r')
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
            time.sleep(self.interval)
            with lock:
                self.create_plot()

    def create_plot(self):
        print(self.__class__.__name__ + ": updating plot.")
        timestamp = time.time()
        now = datetime.utcnow()
        interval = self.interval
        n_events = sum(t > timestamp - interval for t in self.event_times)
        rate = n_events / interval
        self.trigger_rates.append((now, rate))
        try:
            self.store.append(
                'trigger_rates',
                pd.DataFrame({
                    'timestamp': [now],
                    'rate': [rate]
                })
            )
        except pd.io.pytables.ClosedFileError:
            pass
        print(
            "Number of rates recorded: {0} (last: {1}".format(
                len(self.trigger_rates), self.trigger_rates[-1]
            )
        )

        x, y = zip(*self.trigger_rates)
        if not any(y):
            print("Waiting for more values...")
            return
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.xaxis.set_major_formatter(xfmt)
        data = pd.DataFrame({'dates': x, 'rates': y})
        #            plt.scatter(x, y)
        data.plot('dates', 'rates', grid=True, ax=ax, legend=False, style='.')
        ax.set_title(
            "Trigger Rate - via Event Times\n{0}".format(
                datetime.utcnow().strftime("%c")
            )
        )
        ax.set_xlabel("time")
        ax.set_ylabel("trigger rate [Hz]")
        #        ax.set_ylim(-0.1)
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
        if self.thread is not None:
            self.thread.stop()
        if self.store.is_open:
            self.store.close()


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
            time.sleep(50)

    def create_plot(self):
        print(self.__class__.__name__ + ": updating plot.")
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
            # print(self.__class__.__name__ + ": updating plot.")
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
            # ts_series = pd.Series(ts)
            # print(ts_series.describe())

        ax.set_title(
            "DOM Activity - via Summary Slices\n{0}".format(
                datetime.utcnow().strftime("%c")
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


class ZTPlot(Module):
    def configure(self):
        self.run = True
        self.max_queue = 3
        self.queue = Queue()
        self.thread = threading.Thread(target=self.plot).start()

    def process(self, blob):
        if 'Hits' not in blob:
            return blob

        hits = blob['Hits']
        e_info = blob['EventInfo']

        print("Event queue size: {0}".format(self.queue.qsize()))
        if self.queue.qsize() < self.max_queue:
            self.queue.put((e_info, hits))

        return blob

    def plot(self):
        while self.run:
            try:
                e_info, hits = self.queue.get(timeout=50)
            except Empty:
                continue
            with lock:
                self.create_plot(e_info, hits)

    def create_plot(self, e_info, hits):
        print(self.__class__.__name__ + ": updating plot.")
        cal.apply(hits)
        dus = set(detector.doms[h.dom_id][0] for h in hits)

        n_plots = len(dus)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(n_plots / n_cols) + (n_plots % n_cols > 0)
        fig, axes = plt.subplots(
            ncols=n_cols,
            nrows=n_rows,
            sharex=True,
            sharey=True,
            figsize=(16, 8)
        )

        for ax, du in zip(axes.flat, dus):
            _hits = [h for h in hits if detector.doms[h.dom_id][0] == du]
            du_hits = HitSeries(_hits)
            trig_hits = HitSeries([h for h in _hits if h.triggered])

            ax.scatter(
                du_hits.time,
                [z for (x, y, z) in du_hits.pos],
                c='#09A9DE',
                label='hit'
            )
            ax.scatter(
                trig_hits.time,
                [z for (x, y, z) in trig_hits.pos],
                c='#FF6363',
                label='triggered hit'
            )
            ax.set_title('DU{0}'.format(du), fontsize=16, fontweight='bold')

        for ax in axes.flat:
            ax.tick_params(labelsize=16)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
            xlabels = ax.get_xticklabels()
            for label in xlabels:
                label.set_rotation(45)

        plt.suptitle(
            "Run {0}, FrameIndex {1}, TriggerCounter {2}\n{3}".format(
                e_info.run_id, e_info.frame_index, e_info.trigger_counter,
                datetime.utcfromtimestamp(e_info.utc_seconds)
            ),
            fontsize=16
        )
        fig.text(0.5, 0.01, 'time [ns]', ha='center')
        fig.text(0.08, 0.5, 'z [m]', va='center', rotation='vertical')
        #        plt.tight_layout()

        filename = 'ztplot'
        f = os.path.join(PLOTS_PATH, filename + '.png')
        f_tmp = os.path.join(PLOTS_PATH, filename + '_tmp.png')
        plt.savefig(f_tmp, dpi=120, bbox_inches="tight")
        plt.close('all')
        shutil.move(f_tmp, f)

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
pipe.attach(DAQProcessor)
pipe.attach(DOMActivityPlotter)
pipe.attach(TriggerRate)
pipe.attach(DOMHits)
pipe.attach(ZTPlot)
pipe.drain()
