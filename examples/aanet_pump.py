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
pipeline.attach(AanetPump, 'aanet_pump',
                filename='/sps/km3net/users/tgal/data/km3net/mu_oct14/km3net_jul13_90m_muatm10T23.km3_v5r1.JTE.root.aa.root')
#pipeline.attach(DAQEventPrinter, 'moo')
#pipeline.attach(DAQSummaryslicePrinter, 'summaryslice_printer')
#pipeline.attach(MeanRates, 'mean_rates')
pipeline.attach(PrintBlob, 'print_blob')
pipeline.drain(1)


