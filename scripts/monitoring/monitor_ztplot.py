#!/usr/bin/env python
# coding=utf-8
# vim: ts=4 sw=4 et
from __future__ import division

from datetime import datetime
from collections import deque, defaultdict
from functools import partial
import math
import os
import shutil
import time
import threading

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as md
from matplotlib.colors import LogNorm
from matplotlib import pylab
import seaborn as sns
import pandas as pd
import numpy as np

from km3pipe import Pipeline, Module, Geometry
from km3pipe.dataclasses import HitSeries
from km3pipe.common import StringIO, Queue, Empty
from km3pipe.hardware import Detector
from km3pipe.io import CHPump
from km3pipe.io.daq import (DAQProcessor, DAQPreamble, DAQSummaryslice,
                               DAQEvent)
from km3pipe.time import tai_timestamp
import km3pipe.style

from km3pipe.logger import logging

# for logger_name, logger in logging.Logger.manager.loggerDict.iteritems():
#     if logger_name.startswith('km3pipe.'):
#         print("Setting log level to debug for '{0}'".format(logger_name))
#         logger.setLevel("DEBUG")


PLOTS_PATH = 'www/plots'
N_DOMS = 18
N_DUS = 2 
geometry = Geometry(det_id=14)
detector = geometry.detector

xfmt = md.DateFormatter('%Y-%m-%d %H:%M')
lock = threading.Lock()



class ZTPlot(Module):
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.run = True
        self.max_queue = 3
        self.queue = Queue()
        self.thread = threading.Thread(target=self.plot).start()

    def process(self, blob):
        if 'Hits' not in blob:
            return blob

        hits = blob['Hits'].serialise(to='pandas')
        geometry.apply(hits)
        event_info = blob['EventInfo']

        if len(np.unique(hits[hits.triggered == True].du)) < 2:
            print("Skipping...")
            return blob

        print("OK")
        # print("Event queue size: {0}".format(self.queue.qsize()))
        if self.queue.qsize() < self.max_queue:
            self.queue.put((event_info, hits))
        
        return blob

    def plot(self):
        while self.run:
            try:
                event_info, hits = self.queue.get(timeout=50)
            except Empty:
                continue
            with lock:
                self.create_plot(event_info, hits)

    def create_plot(self, event_info, hits):
        print(self.__class__.__name__ + ": updating plot.")
        dus = set(hits.du)

        n_plots = len(dus)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(n_plots / n_cols) + (n_plots % n_cols > 0)
        fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows,
                                 sharex=True, sharey=True, figsize=(16, 8))

        # [axes.flat[-1 - i].axis('off') for i in range(len(axes.flat) - n_plots)]

        for ax, (du, du_hits) in zip(axes.flat, hits.groupby("du")):
            #_hits = [h for h in hits
            #         if detector.doms[h.dom_id][0] == du]
            #trig_hits = HitSeries([h for h in _hits if h.triggered])
            trig_hits = du_hits[du_hits.triggered == True]

            ax.scatter(du_hits.time, du_hits.pos_z,
                       c='#09A9DE', label='hit')
            ax.scatter(trig_hits.time, trig_hits.pos_z,
                       c='#FF6363', label='triggered hit')
            ax.set_title('DU{0}'.format(du), fontsize=16, fontweight='bold')

        for ax in axes.flat:
            ax.tick_params(labelsize=16)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
            xlabels = ax.get_xticklabels()
            for label in xlabels:
                label.set_rotation(45)

        plt.suptitle("FrameIndex {0}, TriggerCounter {1}\n{2} UTC"
                     .format(event_info.frame_index,
                             event_info.trigger_counter,
                             datetime.utcfromtimestamp(event_info.utc_seconds)),
                     fontsize=16)
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
pipe.attach(CHPump, host='127.0.0.1',
                    port=5553,
                    tags='IO_EVT, IO_SUM',
                    timeout=60*60*24*7,
                    max_queue=2000)
pipe.attach(DAQProcessor)
pipe.attach(ZTPlot)
pipe.drain()
