#!/usr/bin/env python
import time

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
        self.filenumber = int(time.time())

    def process(self, blob):
        if 'CHData' in blob:
            tag = str(blob['CHPrefix'].tag)
            data = blob['CHData']
            self.dump(data, tag)
        return blob

    def dump(self, data, tag):
        with open('data/{0}_{1}.dat'.format(tag, self.filenumber), 'a+') as f:
            self.counter += 1
            if self.counter % 10000 == 0:
                self.filenumber = int(time.time())
            f.write(data)


pipe = Pipeline()
pipe.attach(
    CHPump,
    host='127.0.0.1',
    port=5553,
    tags='IO_EVT',
    timeout=60 * 60 * 24,
    max_queue=10
)
pipe.attach(CHPrinter)
pipe.attach(Dumper)
pipe.drain()
