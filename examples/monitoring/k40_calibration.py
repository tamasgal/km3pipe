#!/usr/bin/env python
# coding=utf-8
# vim: ts=4 sw=4 et
"""
=========================
K40 Intra-DOM Calibration
=========================

The following script calculates the PMT time offsets using K40 coincidences

"""
# Author: Jonas Reubelt <jreubelt@km3net.de> and Tamas Gal <tgal@km3net.de>
# License: MIT
import km3pipe as kp
from km3pipe.io.daq import TMCHData
from km3pipe.logger import logging
from km3modules import k40
from km3modules.common import StatusBar, MemoryObserver
import matplotlib.pylab as plt
from collections import defaultdict
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
import io
import time
import km3pipe.style
km3pipe.style.use("km3pipe")

db = kp.db.DBManager()

log_core = kp.logger.get("km3pipe.core")
#log_core.setLevel("DEBUG")


class IntraDOMCalibrationPlotter(kp.Module):
    def process(self, blob):
        calibration = blob["IntraDOMCalibration"]
        #for process in (self.create_plot, self.save_hdf5):
        #for process in (self.create_plot, self.save_hdf5):
        proc = mp.Process(target=self.create_plot, args=(calibration,))
        proc.daemon = True
        proc.start()
        proc.join()

    def create_plot(self, calibration):
        print("Creating plot...")
        fig, axes = plt.subplots(8, 4, figsize=(16, 20),
                                 sharex=True, sharey=True)
        for ax, (dom_id, calib) in zip(axes.flatten(), calibration.items()):
            ax.plot(np.cos(calib['angles']), calib["means"], '.')
            ax.plot(np.cos(calib['angles']), calib["corrected_means"], '.')
            ax.set_title("{0} - {1}".format(db.doms.via_dom_id(dom_id), dom_id))
            ax.set_ylim((-20, 20))
        plt.suptitle("{0} UTC".format(datetime.utcnow().strftime("%c")))
        plt.savefig("intradom.png", bbox_inches='tight')
        plt.close('all')

        fig, axes = plt.subplots(8, 4, figsize=(16, 20),
                                 sharex=True, sharey=True)
        for ax, (dom_id, calib) in zip(axes.flatten(), calibration.items()):
            ax.plot(np.cos(calib['angles']), calib["rates"], '.')
            ax.plot(np.cos(calib['angles']), calib["corrected_rates"], '.')
            ax.set_title("{0} - {1}".format(db.doms.via_dom_id(dom_id), dom_id))
            ax.set_ylim((0, 15))
        plt.suptitle("{0} UTC".format(datetime.utcnow().strftime("%c")))
        plt.savefig("angular_k40rate_distribution.png",
                    bbox_inches='tight')
        plt.close('all')

    def save_hdf5(self, calibration):
        print("Saving calibration information...")
        store = pd.HDFStore('data/k40calib.h5', mode='r')
        now = int(time.time())
        timestamps = (now,) * 31
        for dom_id, calib in calibration.items():
            tdc_channels = range(31)
            t0s = calib['opt_t0s'].x
            dom_ids = (dom_id,) * 31
            df = pd.DataFrame({
                'timestamp': timestamps,
                'dom_id': dom_ids,
                'tdc_channel': tdc_channels,
                't0s': t0s
                })
            store.append('t0s', df, format='table', data_columns=True)
        store.close()


def CHTagger(blob):
    tag = str(blob['CHPrefix'].tag)
    blob[tag] = True
    return blob



class MeanPMTRatesService(kp.Module):
    def configure(self):
        self.mean_rates = {}
        self.rates = defaultdict(lambda: defaultdict(list))
        self.expose(self.mean_rates, 'MeanPMTRates')
        self.index = 0

    def process(self, blob):
        self.index += 1
        tmch_data = TMCHData(io.BytesIO(blob['CHData']))
        dom_id = tmch_data.dom_id
        for channel_id, rate in enumerate(tmch_data.pmt_rates):
            self.rates[dom_id][channel_id].append(rate)

        if self.index % 10*31 == 0:
            self._calculate_mean_rates()
        return blob

    def _calculate_mean_rates(self):
        for dom_id in self.rates.keys():
            self.mean_rates[dom_id] = [np.mean(self.rates[dom_id][c])  \
                                       for c in range(31)]


pipe = kp.Pipeline()
pipe.attach(kp.io.ch.CHPump,
            host='127.0.0.1',
            port=5553,
            tags='IO_TSL, IO_MONIT',
            timeout=7*60*60*24,
            max_queue=42)
pipe.attach(CHTagger)
pipe.attach(StatusBar, every=1000)
pipe.attach(MemoryObserver, every=5000)
pipe.attach(MeanPMTRatesService, only_if='IO_MONIT')
pipe.attach(kp.io.daq.TimesliceParser, only_if='IO_TSL')
pipe.attach(k40.CoincidenceFinder, accumulate=10*20, only_if='TimesliceFrames')
pipe.attach(k40.K40BackgroundSubtractor, only_if='K40Counts')
pipe.attach(k40.IntraDOMCalibrator, only_if='K40Counts')
pipe.attach(IntraDOMCalibrationPlotter, only_if='IntraDOMCalibration')
pipe.drain()
