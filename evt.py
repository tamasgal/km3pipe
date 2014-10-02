__author__ = 'tamasgal'

from km3pipe import Module, Pipeline
from km3pipe.pumps import EvtPump

class PrintBlob(Module):
    def process(self, blob):
        print(blob.keys())
        return blob

pipeline = Pipeline(cycles=1)
pipeline.attach(EvtPump, 'evtpump', filename='example_numuNC.evt')
pipeline.attach(PrintBlob, 'printer')
pipeline.drain()

