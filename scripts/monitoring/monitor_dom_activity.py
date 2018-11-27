#!/usr/bin/env python
# vim: ts=4 sw=4 et

from collections import deque, defaultdict
from functools import partial
from io import BytesIO
import os

import km3pipe as kp
from km3modules.plot import plot_dom_parameters
import km3pipe.style

km3pipe.style.use('km3pipe')

PLOTS_PATH = 'km3web/plots'
cal = kp.calib.Calibration(det_id=29)
detector = cal.detector


class DOMActivityPlotter(kp.Module):
    def configure(self):
        self.index = 0
        self.last_activity = defaultdict(partial(deque, maxlen=4000))
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
        preamble = kp.io.daq.DAQPreamble(file_obj=data_io)    # noqa
        summaryslice = kp.io.daq.DAQSummaryslice(file_obj=data_io)
        timestamp = summaryslice.header.time_stamp

        for dom_id, _ in summaryslice.summary_frames.items():
            du, dom, _ = detector.doms[dom_id]
            self.last_activity[(du, dom)] = timestamp

        self.cuckoo.msg()

        return blob

    def create_plot(self):
        print(self.__class__.__name__ + ": updating plot.")
        filename = os.path.join(PLOTS_PATH, 'dom_activity.png')
        now = kp.time.tai_timestamp()
        delta_ts = {}
        for key, timestamp in self.last_activity.items():
            delta_ts[key] = now - timestamp
        plot_dom_parameters(
            delta_ts,
            detector,
            filename,
            'last activity [s]',
            "DOM Activity - via Summary Slices",
            vmin=0.0,
            vmax=15 * 60
        )


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
pipe.attach(DOMActivityPlotter)
pipe.drain()
