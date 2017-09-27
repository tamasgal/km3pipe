#!/usr/bin/env python
# Author: Tamas Gal <tgal@km3net.de>
# License: MIT
# coding=utf-8
# vim: ts=4 sw=4 et
"""
PMT rates.


Usage:
    pmt_rates.py
    pmt_rates.py --version

Option:
    -h --help       Show this screen.
"""
from datetime import datetime
import io
from collections import defaultdict
import threading
import time
import km3pipe as kp
from km3pipe.io.daq import TMCHData
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import km3pipe.style as kpst
kpst.use("km3pipe")

__author__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
VERSION = "1.0"

log = kp.logger.logging.getLogger("PMTrates")


class PMTRates(kp.Module):
    def configure(self):
        self.detector = self.require("detector")
        self.du = self.require("du")
        self.interval = self.get("interval") or 10
        self.plot_path = self.get("plot_path") or "km3web/plots/pmtrates.png"
        self.max_x = 800
        self.index = 0
        self.rates = defaultdict(list)
        self.rates_matrix = np.full((18*31, self.max_x), np.nan)
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        interval = self.interval
        while True:
            time.sleep(interval)
            now = datetime.now()
            self.add_column()
            self.update_plot()
            with self.lock:
                self.rates = defaultdict(list)
            delta_t = (datetime.now() - now).total_seconds()
            remaining_t = self.interval - delta_t
            log.info("Delta t: {} -> waiting for {}s"
                     .format(delta_t, self.interval - delta_t))
            if(remaining_t < 0):
                log.error("Can't keep up with plot production. "
                          "Increase the interval!")
                interval = 1
            else:
                interval = remaining_t

    def add_column(self):
        m = np.roll(self.rates_matrix, -1, 1)
        y_range = 18*31
        mean_rates = np.full(y_range, np.nan)
        for i in range(y_range):
            if i not in self.rates:
                continue
            mean_rates[i] = np.mean(self.rates[i])

        m[:, self.max_x - 1] = mean_rates
        self.rates_matrix = m

    def update_plot(self):
        print("Updating plot at {}".format(self.plot_path))
        m = self.rates_matrix
        m[m > 15000] = 15000
        m[m < 5000] = 5000
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(m, origin='lower')
        ax.set_title("Mean PMT Rates for DU-{} (colours from 5kHz to 15kHz)\n{}"
                     .format(self.du, datetime.utcnow()))
        ax.set_xlabel("time interval [{}s/px] (latest on the right)"
                      .format(self.interval))
        plt.yticks([i*31 for i in range(18)],
                   ["Floor {}".format(f) for f in range(1, 19)])
        fig.tight_layout()
        plt.savefig(self.plot_path)
        plt.close('all')

    def process(self, blob):
        tmch_data = TMCHData(io.BytesIO(blob['CHData']))
        dom_id = tmch_data.dom_id

        if dom_id not in self.detector.doms:
            return blob

        du, floor, _ = self.detector.doms[dom_id]

        if du != self.du:
            return blob

        y_base = (floor - 1) * 31

        for channel_id, rate in enumerate(tmch_data.pmt_rates):
            idx = y_base + channel_id
            with self.lock:
                self.rates[idx].append(rate)

        return blob


def main():
    from docopt import docopt
    args = docopt(__doc__, version=VERSION)

    detector = kp.hardware.Detector(det_id=29)
    pipe = kp.Pipeline(timeit=True)
    pipe.attach(kp.io.CHPump,
                host='192.168.0.110',
                port=5553,
                tags='IO_MONIT',
                timeout=60*60*24*7,
                max_queue=1000)
    pipe.attach(PMTRates, detector=detector, du=2, interval=2)
    pipe.drain()


if __name__ == "__main__":
    main()