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
import matplotlib
matplotlib.use('Agg')
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
PLOTS_PATH = "km3web/plots"

class IntraDOMCalibrationPlotter(kp.Module):
    def process(self, blob):
        calibration = blob["IntraDOMCalibration"]
        for process in (self.create_plot, self.save_hdf5):
            proc = mp.Process(target=process, args=(calibration,))
            proc.daemon = True
            proc.start()
            proc.join()

    def create_plot(self, calibration):
        print("Creating plot...")
        fig, axes = plt.subplots(6, 3, figsize=(16, 20),
                                 sharex=True, sharey=True)
        for ax, (dom_id, calib) in zip(axes.flatten(), calibration.items()):
            ax.plot(np.cos(calib['angles']), calib["means"], '.')
            ax.plot(np.cos(calib['angles']), calib["corrected_means"], '.')
            ax.set_title("{0} - {1}".format(db.doms.via_dom_id(dom_id), dom_id))
            ax.set_ylim((-20, 20))
        plt.suptitle("{0} UTC".format(datetime.utcnow().strftime("%c")))
        plt.savefig(PLOTS_PATH + "/intradom.png", bbox_inches='tight')
        plt.close('all')

        fig, axes = plt.subplots(6, 3, figsize=(16, 20),
                                 sharex=True, sharey=True)
        for ax, (dom_id, calib) in zip(axes.flatten(), calibration.items()):
            ax.plot(np.cos(calib['angles']), calib["rates"], '.')
            ax.plot(np.cos(calib['angles']), calib["corrected_rates"], '.')
            ax.set_title("{0} - {1}".format(db.doms.via_dom_id(dom_id), dom_id))
            ax.set_ylim((0, 10))
        plt.suptitle("{0} UTC".format(datetime.utcnow().strftime("%c")))
        plt.savefig(PLOTS_PATH + "/angular_k40rate_distribution.png",
                    bbox_inches='tight')
        plt.close('all')

    def save_hdf5(self, calibration):
        print("Saving calibration information...")
        store = pd.HDFStore('data/k40calib.h5', mode='a')
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



class MedianPMTRatesService(kp.Module):
    def configure(self):
        self.rates = defaultdict(lambda: defaultdict(list))
        self.expose(self.get_median_rates, 'GetMedianPMTRates')

    def process(self, blob):
        tmch_data = TMCHData(io.BytesIO(blob['CHData']))
        dom_id = tmch_data.dom_id
        for channel_id, rate in enumerate(tmch_data.pmt_rates):
            self.rates[dom_id][channel_id].append(rate)
        return blob

    def get_median_rates(self):
        print("Calculating median PMT rates.")
        median_rates = {}
        for dom_id in self.rates.keys():
            median_rates[dom_id] = [np.median(self.rates[dom_id][c])  \
                                  for c in range(31)]
        self.rates = defaultdict(lambda: defaultdict(list))
        return median_rates


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
pipe.attach(MedianPMTRatesService, only_if='IO_MONIT')
pipe.attach(kp.io.daq.TimesliceParser, only_if='IO_TSL')
pipe.attach(k40.CoincidenceFinder,
            accumulate=10*60*20,
            only_if='TimesliceFrames',
            tmax=10)
pipe.attach(k40.K40BackgroundSubtractor, only_if='K40Counts')
pipe.attach(k40.IntraDOMCalibrator, only_if='K40Counts', fit_background=False,
            ctmin=0.)
pipe.attach(IntraDOMCalibrationPlotter, only_if='IntraDOMCalibration')
pipe.drain()
