#!/usr/bin/env python
# coding=utf-8
# vim: ts=4 sw=4 et
from __future__ import division

from collections import deque, defaultdict
from functools import partial
from io import BytesIO
import os

import numpy as np

import km3pipe as kp
from km3modules.plot import plot_dom_parameters
import km3pipe.style

from km3pipe.logger import logging

# for logger_name, logger in logging.Logger.manager.loggerDict.iteritems():
#     if logger_name.startswith('km3pipe.'):
#         print("Setting log level to debug for '{0}'".format(logger_name))
#         logger.setLevel("DEBUG")


PLOTS_PATH = 'www/plots'
geometry = kp.core.Geometry(det_id=14)
detector = geometry.detector


class DOMActivityPlotter(kp.Module):
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.index = 0
        self.rates = defaultdict(partial(deque, maxlen=4000))
        self.cuckoo = kp.time.Cuckoo(60, self.create_plot)

    def process(self, blob):
        self.index += 1
        if self.index % 30:
            return blob

        tag = str(blob['CHPrefix'].tag)

        if not tag == 'IO_SUM':
            return blob

        data = blob['CHData']
        data_io = BytesIO(data)
        preamble = kp.io.daq.DAQPreamble(file_obj=data_io)
        summaryslice = kp.io.daq.DAQSummaryslice(file_obj=data_io)
        timestamp = summaryslice.header.time_stamp
        now = kp.time.tai_timestamp()
        for dom_id, rates in summaryslice.summary_frames.iteritems():
            du, dom, _ = detector.doms[dom_id]
            self.rates[(du, dom)] = now - timestamp

        self.cuckoo.msg()

        return blob

    def create_plot(self):
        print(self.__class__.__name__ + ": updating plot.")
        filename = os.path.join(PLOTS_PATH, 'dom_activity.png')
        plot_dom_parameters(self.rates, detector, filename,
                            'last activity [s]',
                            "DOM Activity - via Summary Slices",
                            vmin=0.0, vmax=15*60)


pipe = kp.Pipeline()
pipe.attach(kp.io.ch.CHPump, host='127.0.0.1',
            port=5553,
            tags='IO_SUM',
            timeout=60*60*24*7,
            max_queue=2000)
pipe.attach(kp.io.daq.DAQProcessor)
pipe.attach(DOMActivityPlotter)
pipe.drain()
