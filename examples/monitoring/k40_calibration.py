#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from km3modules import k40
from km3modules.common import StatusBar, MemoryObserver, Siphon
from km3modules.plot import IntraDOMCalibrationPlotter
import km3pipe.style

km3pipe.style.use("km3pipe")

pipe = kp.Pipeline(timeit=True)
pipe.attach(
    kp.io.ch.CHPump,
    host="127.0.0.1",
    port=5553,
    tags="IO_TSL, IO_MONIT",
    timeout=7 * 60 * 60 * 24,
    max_queue=42,
)
pipe.attach(kp.io.ch.CHTagger)
pipe.attach(StatusBar, every=1000)
pipe.attach(MemoryObserver, every=5000)
pipe.attach(k40.MedianPMTRatesService, only_if="IO_MONIT")
pipe.attach(kp.io.daq.TimesliceParser)
pipe.attach(k40.TwofoldCounter, tmax=10)
pipe.attach(Siphon, volume=10 * 10 * 1, flush=True)
pipe.attach(k40.K40BackgroundSubtractor)
pipe.attach(k40.IntraDOMCalibrator, ctmin=0.0)
pipe.attach(IntraDOMCalibrationPlotter)
pipe.attach(k40.ResetTwofoldCounts)
pipe.drain()
