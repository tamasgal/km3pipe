#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: ts=4 sw=4 et
"""
========
HDF5Sink
========

This examples show how to create dummy (event) data, dump them to
an HDF5 file using the ``kp.io.HDF5Sink`` module and how to read them back
with pandas or the ``kp.io.HDF5Pump`` module.

"""

import km3pipe as kp
import pandas as pd
import numpy as np

filename = "reco_results.h5"

####################################################################
# The ``DataGenerator`` is a dumm pump which creates random event
# data and a common header.


class DataGenerator(kp.Module):
    def configure(self):
        self.header = kp.Table({'is_mc': True},
                               h5singleton=True,
                               h5loc='/header')

    def process(self, blob):

        n = np.random.randint(2, 20)
        tab = kp.Table({
            'lambda': np.random.rand(n),
            'energy': np.random.rand(n) * 1000
        },
                       h5loc='/reco',
                       name='Quality Parameter')
        event_info = kp.Table({"run_id": np.random.randint(10000, 20000)},
                              h5loc='/event_info')
        blob['Reco'] = tab
        blob['EventInfo'] = event_info
        blob['Header'] = self.header
        return blob


####################################################################
# The ``printer`` just prints the blob, obviously.
# data and a common header.


def printer(blob):
    print(blob)
    return blob


####################################################################
# The following pipeline will do 10 iterations and dump the data
# properly formatted into our output file, ``reso_results.h5``

pipe = kp.Pipeline()
pipe.attach(DataGenerator)
pipe.attach(printer)
pipe.attach(kp.io.HDF5Sink, filename=filename)
pipe.drain(10)

####################################################################
# Pandas offers an easy way to read the generated data. As you can
# see, no km3pipe is needed, the only requirement is ``hdf5lib``
# and of course some kind of wrapper library for easy acces, in this case
# Pandas but you can use h5py, pytables directly or other languages.

header = pd.read_hdf(filename, "/header")
event_info = pd.read_hdf(filename, "/event_info")
reco = pd.read_hdf(filename, "/reco")
print(reco.head(3))

####################################################################
# This last section shows how to read the created data with km3pipe
# pipelines.


def inspector(blob):
    reco = blob['Reco']
    lambdas = reco['lambda']
    print(lambdas, type(lambdas), lambdas.dtype)
    return blob


pipe = kp.Pipeline()
pipe.attach(kp.io.HDF5Pump, filename=filename)
pipe.attach(printer)
pipe.attach(inspector)
pipe.drain()
