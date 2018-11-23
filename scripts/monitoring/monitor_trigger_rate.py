#!/usr/bin/env python
# vim: ts=4 sw=4 et

from datetime import datetime
from collections import deque
import sys
from io import StringIO
import os
import shutil
import time
import threading

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')    # noqa
import matplotlib.pyplot as plt
import matplotlib.dates as md
import pandas as pd

from km3pipe import Pipeline, Module
from km3pipe.config import Config
from km3pipe.io import CHPump
from km3pipe.io.daq import (DAQPreamble, DAQEvent)
from km3pipe.logger import get_logger
import km3pipe.style

km3pipe.style.use('km3pipe')
log = get_logger("trigger_rate")

PLOTS_PATH = 'www/plots'

xfmt = md.DateFormatter('%Y-%m-%d %H:%M')
lock = threading.Lock()


def trigger_rate_sampling_period():
    try:
        return int(Config().get("Monitoring", "trigger_rate_sampling_period"))
    except (TypeError, ValueError):
        return 180


class TriggerRate(Module):
    def configure(self):
        self.run = True
        self.interval = self.get("interval") or trigger_rate_sampling_period()
        self.event_times = deque(maxlen=4000)    # max events per interval
        # minutes to monitor
        self.trigger_rates = deque(maxlen=60 * 24 // (self.interval // 60))
        self.thread = threading.Thread(target=self.plot).start()
        self.store = pd.HDFStore('data/trigger_rates.h5')
        self.restore_data()

    def restore_data(self):
        with lock:
            try:
                data = zip(
                    self.store.trigger_rates.timestamp,
                    self.store.trigger_rates.rate
                )
                self.trigger_rates.extend(data)
                print(
                    "\n{0} data points restored.".format(
                        len(self.trigger_rates)
                    )
                )
            except AttributeError:
                pass

    def process(self, blob):
        if not str(blob['CHPrefix'].tag) == 'IO_EVT':
            return blob
        sys.stdout.write('.')
        sys.stdout.flush()

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
            self.interval = trigger_rate_sampling_period()
            time.sleep(self.interval)
            with lock:
                self.create_plot()

    def create_plot(self):
        print('\n' + self.__class__.__name__ + ": updating plot.")
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

        x, y = zip(*self.trigger_rates)
        if not any(y):
            return
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.xaxis.set_major_formatter(xfmt)
        data = pd.DataFrame({'dates': x, 'rates': y})
        data.plot('dates', 'rates', grid=True, ax=ax, legend=False, style='.')
        ax.set_title(
            "Trigger Rate - via Event Times\n{0} UTC".format(
                datetime.utcnow().strftime("%c")
            )
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
        print("Plot updated.")

    def finish(self):
        self.run = False
        if self.thread is not None:
            self.thread.stop()
        if self.store.is_open:
            self.store.close()


pipe = Pipeline()
pipe.attach(CHPump, tags='IO_EVT', timeout=60 * 60 * 24 * 7, max_queue=2000)
pipe.attach(TriggerRate)
pipe.drain()
