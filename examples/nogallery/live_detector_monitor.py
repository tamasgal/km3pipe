#!/usr/bin/env python

from time import ctime

from royweb import PacketHandler

from km3pipe import Pipeline, Module
from km3pipe.io import CHPump
from km3pipe.io.daq import DAQPreamble, DAQEvent
from io import StringIO


class CHPrinter(Module):
    def process(self, blob):
        print("New blob:")
        print(blob['CHPrefix'])
        return blob


class ROySender(Module):
    def configure(self):
        self.packet_handler = PacketHandler('131.188.161.241', 9999)

    def process(self, blob):
        roy = self.packet_handler
        data_size = blob['CHPrefix'].length
        roy.send('evt_size', data_size, 'byte', '')

        data = blob['CHData']
        data_io = StringIO(data)

        preamble = DAQPreamble(file_obj=data_io)
        print(ctime())
        print(preamble)

        event = DAQEvent(file_obj=data_io)
        print("Run: {0}".format(event.header.run))
        print("Time Slice: {0}".format(event.header.time_slice))

        try:
            ratio = event.n_triggered_hits / event.n_snapshot_hits * 100
        except ZeroDivisionError:
            ratio = 100 if event.n_triggered_hits > 1 else 0
        roy.send("hit_ratio", ratio, '%', '')
        roy.send("trig_hits", event.n_triggered_hits, '#', '')
        roy.send("hits", event.n_snapshot_hits, '#', '')
        roy.send("timeslice", event.header.time_slice, '#', '')
        roy.send("run", event.header.run, '#', '')

        doms = {dom_id for (dom_id, _, _, _) in event.snapshot_hits}
        trig_doms = {dom_id for (dom_id, _, _, _, _) in event.triggered_hits}
        roy.send("doms", len(doms), '#', '')
        roy.send("trig_doms", len(trig_doms), '#', '')

        hit_times = [time for (_, _, time, _) in event.snapshot_hits]
        try:
            length = abs(max(hit_times) - min(hit_times))
        except ValueError:
            pass
        else:
            roy.send("evt_length", length, 'ns', '')

        # if event.n_snapshot_hits < 30:
        #    tots = [tot for (_, _, _, tot) in event.snapshot_hits]
        #    for tot in tots[::3]:
        #        roy.send("tot", tot, 'ns', '')

        return blob


pipe = Pipeline()
pipe.attach(
    CHPump,
    host='localhost',
    port=5553,
    tags='IO_EVT',
    timeout=60 * 60 * 24,
    max_queue=50
)
pipe.attach(ROySender)
pipe.attach(CHPrinter)

pipe.drain()
