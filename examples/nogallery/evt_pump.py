__author__ = 'tamasgal'

from km3pipe import Module, Pipeline
from km3pipe.io import EvtPump


class PrintBlob(Module):
    def process(self, blob):
        print(blob.keys())
        return blob


pipeline = Pipeline()
pipeline.attach(EvtPump, 'evtpump', filename='files/example_numuNC.evt')
pipeline.attach(PrintBlob, 'printer')
pipeline.drain(1)
