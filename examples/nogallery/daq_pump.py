from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

from km3pipe import Pipeline, Module
from km3pipe.io import DAQPump


class DAQEventPrinter(Module):
    def process(self, blob):
        try:
            print(blob['DAQEvent'])
        except KeyError:
            pass
        return blob


class DAQSummaryslicePrinter(Module):
    def process(self, blob):
        try:
            print(blob['DAQSummaryslice'])
        except KeyError:
            pass
        return blob


class MeanHits(Module):
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.hits = []

    def process(self, blob):
        try:
            event = blob['DAQEvent']
            self.hits.append(event.n_snapshot_hits)
        except KeyError:
            pass
        return blob

    def finish(self):
        mean_hits = sum(self.hits) / len(self.hits)
        print("Number of entries: {0}\nMean hits: {1}"
              .format(len(self.hits), mean_hits))


class MeanRates(Module):
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.rates = {}

    def process(self, blob):
        try:
            summaryslice = blob['DAQSummaryslice']
            print(summaryslice.summary_frames)
        except KeyError:
            pass
        return blob

    def finish(self):
        pass


pipeline = Pipeline()
pipeline.attach(DAQPump, 'daq_pump',
                filename='/Users/tamasgal/Desktop/RUN-PPM_DU-00430-20140730-121124_detx.dat')
#pipeline.attach(DAQEventPrinter, 'moo')
#pipeline.attach(DAQSummaryslicePrinter, 'summaryslice_printer')
#pipeline.attach(MeanRates, 'mean_rates')
pipeline.attach(MeanHits, 'mean_hits')
pipeline.drain()


