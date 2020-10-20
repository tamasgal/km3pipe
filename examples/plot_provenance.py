#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==========
Provenance
==========

"""

# Author: Tamás Gál <tgal@km3net.de>
# License: MIT
# Date: 2020-08-23

#####################################################
# Introduction
# ------------
# KM3Pipe uses the provenance functionality from ``thepipe``
# which automatically tracks each activity. This document
# shows how it works.

import km3pipe as kp
import km3modules as km
import numpy as np

#####################################################
# Some dummy modules
# ------------------


class RandomNumberGenerator(kp.Module):
    def configure(self):
        self.h5loc = self.require("h5loc")
        self.n = self.get("n", default=10)

    def process(self, blob):
        table = kp.Table({"x": np.random.randn(self.n)}, h5loc=self.h5loc)
        blob["RandomNumbers"] = table
        return blob


#####################################################
# Creating a simple pipeline
# --------------------------
# We create a very basic pipeline:

pipe = kp.Pipeline()
pipe.attach(km.StatusBar, every=1)
pipe.attach(km.mc.GlobalRandomState, seed=23)
pipe.attach(RandomNumberGenerator, h5loc="/rnd", n=5)
pipe.attach(kp.io.HDF5Sink, filename="rnd.h5")
pipe.drain(11)


#####################################################
# Provenance
# ----------
# The provenance information is managed by the singleton class
# ``Provenance``. To access all the provenance information,
# use the ``as_json()`` method:

print(kp.Provenance().as_json(indent=2))
