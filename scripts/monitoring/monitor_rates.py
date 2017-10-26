#!/usr/bin/env python
# coding=utf-8
# vim: ts=4 sw=4 et
from __future__ import division

from collections import deque, defaultdict
from itertools import combinations
from functools import partial
from io import BytesIO
import os

import numpy as np

import km3pipe as kp
from km3modules.plot import plot_dom_parameters
from km3modules.fit import fit_delta_ts
import km3pipe.style

from km3pipe.logger import logging

# for logger_name, logger in logging.Logger.manager.loggerDict.iteritems():
#     if logger_name.startswith('km3pipe.'):
#         print("Setting log level to debug for '{0}'".format(logger_name))
#         logger.setLevel("DEBUG")


PLOTS_PATH = 'www/plots'
geometry = kp.core.Geometry(det_id=14)
detector = geometry.detector


class MonitorRates(kp.Module):
    def configure(self):
        self.index = 0
        self.k40_2fold = {}
        self.rates = {}
        self.cuckoo = kp.time.Cuckoo(60, self.create_plot)
        self.n_slices = 0

    def process(self, blob):
        self.index += 1
        if self.index % 30:
            return blob

        data = blob['CHData']
        data_io = BytesIO(data)
        preamble = kp.io.daq.DAQPreamble(file_obj=data_io)
        summaryslice = kp.io.daq.DAQSummaryslice(file_obj=data_io)
        timestamp = summaryslice.header.time_stamp
        now = kp.time.tai_timestamp()
        for dom_id, rates in summaryslice.summary_frames.iteritems():
            du, dom, _ = detector.doms[dom_id]
            self.rates[(du, dom)] = np.sum(rates) / 1000

        self.cuckoo.msg()

        return blob

    def create_plot(self):
        print(self.__class__.__name__ + ": updating plot.")

        filename = os.path.join(PLOTS_PATH, 'dom_rates.png')
        plot_dom_parameters(self.rates, detector, filename,
                            'rate [kHz]',
                            "DOM Rates",
                            vmin=190, vmax=230,
                            cmap='coolwarm', missing='black',
                            under='darkorchid', over='deeppink')
        print("done")


pipe = kp.Pipeline()
pipe.attach(kp.io.ch.CHPump, host='127.0.0.1',
            port=5553,
            tags='IO_SUM',
            timeout=60*60*24*7,
            max_queue=2000)
pipe.attach(kp.io.daq.DAQProcessor)
pipe.attach(MonitorRates)
pipe.drain()
