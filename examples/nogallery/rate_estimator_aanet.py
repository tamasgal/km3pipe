"""Open a file with real data and estimate the DOM rates.
"""

from km3pipe import Module, Pipeline
from km3pipe.io.aanet import AanetPump


class RateEstimator(Module):
    def process(self, blob):
        hits = blob["Hits"]
        doms = {h.dom_id for h in hits}
        hit_times = [h.t for h in hits]
        event_length = max(hit_times) - min(hit_times)
        n_doms = len(doms)
        n_hits = len(hits)

        print("Active DOMs: {0}".format(doms))
        print("Number of active DOMs: {0}".format(n_doms))
        print("Event length: {0} ns".format(event_length))
        print("Number of hits: {0}".format(n_hits))

        rate_per_dom = 1e6 / (event_length / (n_hits / n_doms))

        print("Estimated rate per DOM: {0} kHz".format(rate_per_dom))


pipe = Pipeline()
pipe.attach(AanetPump, filename='KM3NeT_00000007_00001000.root')
pipe.attach(RateEstimator)
pipe.drain(10)
