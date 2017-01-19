"""Read & dump events through the CH Pump.
"""
from collections import defaultdict
from itertools import combinations
from struct import unpack
from io import BytesIO
from time import sleep

import numpy as np

from km3pipe import Pipeline, Module
from km3pipe.io.ch import CHPump
from km3pipe.logger import logging

#log_ch = logging.getLogger('km3pipe.io.ch')
#log_ch.setLevel('DEBUG')

class TimesliceParser(Module):
    def process(self, blob):
        data = BytesIO(blob['CHData'])
        tsl_size, datatype = unpack('<ii', data.read(8))
        det_id, run, sqnr = unpack('<iii', data.read(12))
        timestamp, ns_ticks, n_frames = unpack('<iii', data.read(12))

#        print("Det ID: {0}, Run: {1}, Sequence Nr.: {2}, Frames: {3}"
#              .format(det_id, run, sqnr, n_frames))

        ts_frames = blob['TimesliceFrames'] = defaultdict(list)

        for i in range(n_frames):
            frame_size, datatype = unpack('<ii', data.read(8))
            det_id, run, sqnr = unpack('<iii', data.read(12))
            timestamp, ns_ticks, dom_id = unpack('<iii', data.read(12))
            dom_status = unpack('<iiiii', data.read(5*4))
            n_hits = unpack('<i', data.read(4))[0]
            hits = []
            for j in range(n_hits):
                hit = unpack('!BlB', data.read(6))
                ts_frames[dom_id].append(hit)
        return blob


def printer(blob):
    print(blob['TimesliceFrames'][808447091])
    return blob


def mongincidence(times, tdcs, tmax=20):
    coincidences = []
    cur_t = 0
    las_t = 0
    for t_idx, t in enumerate(times):
        cur_t = t
        diff = cur_t - las_t
        if diff <= tmax and t_idx > 0 and tdcs[t_idx - 1] != tdcs[t_idx]:
            coincidences.append(((tdcs[t_idx - 1], tdcs[t_idx]), diff))
        las_t = cur_t
    return coincidences


class CoincidenceFinder(Module):
    def configure(self):
        self.m = np.zeros(shape=(465, 41))

    def process(self, blob):
        hits = blob['TimesliceFrames'][808447091]
        hits.sort(key=lambda x: x[1])
        coinces = mongincidence([t for (_,t,_) in hits], [t for (t,_,_) in hits])

        combs = list(combinations(range(31), 2))
        for pmt_pair, t in coinces:
            if pmt_pair[0] > pmt_pair[1]:
                pmt_pair = (pmt_pair[1], pmt_pair[0])
                t = -t
            self.m[combs.index(pmt_pair), t+20] += 1

        print(self.m)

        return blob

    def finish(self):
        np.savetxt("coincidences.txt", self.m)


class Dumper(Module):
    def configure(self):
        self.counter = 0

    def process(self, blob):
        print("New blob:")
        print(blob['CHPrefix'])
        if 'CHData' in blob:
            tag = str(blob['CHPrefix'].tag)
            data = blob['CHData']
            self.dump(data, tag)
        return blob

    def dump(self, data, tag):
        with open('{0}-{1:06}.dat'.format(tag, self.counter), 'bw') as f:
            self.counter += 1
            f.write(data)


pipe = Pipeline()
pipe.attach(CHPump, host='172.16.65.58',
            port=5553,
            tags='IO_TSL',
            timeout=60*60*24,
            max_queue=42)
#pipe.attach(Dumper)
pipe.attach(TimesliceParser)
#pipe.attach(printer)
pipe.attach(CoincidenceFinder)
pipe.drain()
