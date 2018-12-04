# -*- coding: utf-8 -*-
"""Read & dump events through the CH Pump.

"""
from __future__ import absolute_import, print_function, division

from km3pipe import Pipeline, Module
from km3pipe.io import CHPump


class CHPrinter(Module):
    def process(self, blob):
        print("New blob:")
        print(blob['CHPrefix'])
        return blob


class Dumper(Module):
    def configure(self):
        self.counter = 0

    def process(self, blob):
        if 'CHData' in blob:
            tag = str(blob['CHPrefix'].tag)
            data = blob['CHData']
            self.dump(data, tag)
        return blob

    def dump(self, data, tag):
        with open('{0}-{1:06}.dat'.format(tag, self.counter), 'w') as f:
            self.counter += 1
            f.write(data)


pipe = Pipeline()
pipe.attach(
    CHPump,
    host='127.0.0.1',
    port=5553,
    tags='IO_EVT, IO_TSL, IO_SUM, TRG_PARS',
    timeout=60 * 60 * 24,
    max_queue=42
)
pipe.attach(CHPrinter)
pipe.attach(Dumper)
pipe.drain()
