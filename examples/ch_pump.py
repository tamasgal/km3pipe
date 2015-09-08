import time

from km3pipe import Pipeline, Module
from km3pipe.pumps import CHPump

class CHPrinter(Module):
    def process(self, blob):
        print blob['CHPrefix']
        print blob['CHData']
        return blob

pipe = Pipeline()
pipe.attach(CHPump, host='127.0.0.1', port=5553, tag='foo')
pipe.attach(CHPrinter)
pipe.drain(10)

