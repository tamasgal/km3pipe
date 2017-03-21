#!/usr/bin/env python
# coding=utf-8
# vim: ts=4 sw=4 et
"""
=================
Live Event Dumper
=================

Recieves triggered events from the detector and dumps them to a file.

"""
# Author: Tamas Gal <tgal@km3net.de>
# License: MIT
"""Read & dump events through the CH Pump.
"""
from km3pipe import Pipeline, Module
from km3pipe.io.ch import CHPump
from km3pipe.io.daq import DAQProcessor


class Dumper(Module):
    def configure(self):
        self.counter = 0

    def process(self, blob):
	tag = str(blob['CHPrefix'].tag)
	data = blob['CHData']
	self.dump(data, tag)
	print(blob["Hits"])
        return blob

    def dump(self, data, tag):
        with open('{0}-{1:06}.dat'.format(tag, self.counter), 'bw') as f:
            self.counter += 1
            f.write(data)


pipe = Pipeline()
pipe.attach(CHPump, host='xxx.xxx.xxx.xxx',
            port=5553,
            tags='IO_EVT',
            timeout=60*60*24,
            max_queue=42)
pipe.attach(DAQProcessor)
pipe.attach(Dumper)
pipe.drain()
