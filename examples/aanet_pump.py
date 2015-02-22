from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

from km3pipe import Pipeline, Module
from km3pipe.pumps import AanetPump


class PrintBlob(Module):
    def process(self, blob):
        print(blob)
        hit = blob['a_hit']
        print(hit)
        print(hit.t)
        return blob

pipeline = Pipeline()
pipeline.attach(AanetPump, 'aanet_pump', filename='foo.aa.root')
pipeline.attach(PrintBlob, 'print_blob')
pipeline.drain(1)


