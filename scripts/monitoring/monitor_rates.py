#!/usr/bin/env python
# vim: ts=4 sw=4 et

from io import BytesIO
import os

import numpy as np

import km3pipe as kp
from km3modules.plot import plot_dom_parameters
import km3pipe.style

km3pipe.style.use('km3pipe')

PLOTS_PATH = 'www/plots'
cal = kp.calib.Calibration(det_id=14)
detector = cal.detector


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
        preamble = kp.io.daq.DAQPreamble(file_obj=data_io)    # noqa
        summaryslice = kp.io.daq.DAQSummaryslice(file_obj=data_io)
        for dom_id, rates in summaryslice.summary_frames.iteritems():
            du, dom, _ = detector.doms[dom_id]
            self.rates[(du, dom)] = np.sum(rates) / 1000

        self.cuckoo.msg()

        return blob

    def create_plot(self):
        print(self.__class__.__name__ + ": updating plot.")

        filename = os.path.join(PLOTS_PATH, 'dom_rates.png')
        plot_dom_parameters(
            self.rates,
            detector,
            filename,
            'rate [kHz]',
            "DOM Rates",
            vmin=190,
            vmax=230,
            cmap='coolwarm',
            missing='black',
            under='darkorchid',
            over='deeppink'
        )
        print("done")


pipe = kp.Pipeline()
pipe.attach(
    kp.io.ch.CHPump,
    host='127.0.0.1',
    port=5553,
    tags='IO_SUM',
    timeout=60 * 60 * 24 * 7,
    max_queue=2000
)
pipe.attach(kp.io.daq.DAQProcessor)
pipe.attach(MonitorRates)
pipe.drain()
