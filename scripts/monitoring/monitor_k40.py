#!/usr/bin/env python
# vim: ts=4 sw=4 et

from itertools import combinations
import os

import numpy as np

import km3pipe as kp
from km3modules.plot import plot_dom_parameters
from km3modules.fit import fit_delta_ts
import km3pipe.style

km3pipe.style.use('km3pipe')

PLOTS_PATH = 'www/plots'
cal = kp.calib.Calibration(det_id=14)
detector = cal.detector


def mongincidence(times, tdcs, tmax=20, offset=0):
    coincidences = []
    cur_t = 0
    las_t = 0
    for t_idx, t in enumerate(times):
        cur_t = t
        diff = cur_t - las_t
        if offset < diff <= offset + tmax \
                and t_idx > 0 \
                and tdcs[t_idx - 1] != tdcs[t_idx]:
            coincidences.append(((tdcs[t_idx - 1], tdcs[t_idx]), diff))
        las_t = cur_t
    return coincidences


class MonitorK40(kp.Module):
    def configure(self):
        self.index = 0
        self.k40_2fold = {}
        self.rates = {}
        self.cuckoo = kp.time.Cuckoo(300, self.create_plot)
        self.n_slices = 0

    def process(self, blob):
        self.index += 1
        #        if self.index % 30:
        #            return blob

        self.n_slices += 1

        for dom_id, hits in blob['TimesliceFrames'].iteritems():
            du, dom, _ = detector.doms[dom_id]
            omkey = (du, dom)
            if omkey not in self.rates:
                self.rates[omkey] = np.zeros(shape=(465, 41))
            hits.sort(key=lambda x: x[1])
            times = [t for (_, t, _) in hits]
            tdcs = [t for (t, _, _) in hits]
            coinces = mongincidence(times, tdcs)
            combs = list(combinations(range(31), 2))
            for pmt_pair, t in coinces:
                if pmt_pair[0] > pmt_pair[1]:
                    pmt_pair = (pmt_pair[1], pmt_pair[0])
                    t = -t
                self.rates[omkey][combs.index(pmt_pair), t + 20] += 1

        self.cuckoo.msg()

        return blob

    def create_plot(self):
        print(self.__class__.__name__ + ": updating plot.")

        data = {}
        for omkey, m in self.rates.iteritems():
            print("Fitting {0}".format(omkey))
            time = self.n_slices / 10
            rates, means = fit_delta_ts(m, time)
            overall_rate = np.sum(rates)
            print(overall_rate)
            data[omkey] = overall_rate
        self.rates = {}
        self.n_slices = 0

        filename = os.path.join(PLOTS_PATH, 'k40.png')
        plot_dom_parameters(
            data,
            detector,
            filename,
            'rate [Hz]',
            "K40 rate",
            vmin=600,
            vmax=1200,
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
    tags='IO_TSL',
    timeout=60 * 60 * 24 * 7,
    max_queue=2000
)
pipe.attach(kp.io.daq.TimesliceParser)
pipe.attach(MonitorK40)
pipe.drain()
