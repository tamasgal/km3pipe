#!/usr/bin/env python
from time import ctime
from io import StringIO
from pyslack import SlackClient

from km3pipe import Pipeline, Module
from km3pipe.io import CHPump
from km3pipe.io.daq import DAQPreamble, DAQEvent


class CHPrinter(Module):
    def process(self, blob):
        print("New blob:")
        print(blob['CHPrefix'])
        return blob


class SlackSender(Module):
    def configure(self):
        self.client = SlackClient('YOUR_SLACK_API_TOKEN_HERE')
        self.current_run = None

    def process(self, blob):
        data = blob['CHData']
        data_io = StringIO(data)

        preamble = DAQPreamble(file_obj=data_io)
        print(ctime())
        print(preamble)

        event = DAQEvent(file_obj=data_io)
        print("Run: {0}".format(event.header.run))
        print("Time Slice: {0}".format(event.header.time_slice))

        if not self.current_run == event.header.run:
            self.current_run = event.header.run
            self.client.chat_post_message(
                "#du2-live",
                "Run #{0} has started!".format(event.header.run),
                username="slackbot"
            )

        return blob


pipe = Pipeline()
pipe.attach(
    CHPump,
    host='127.0.0.1',    # You'll need an SSH tunnel
    port=5553,
    tags='IO_EVT',
    timeout=60 * 60 * 24,
    max_queue=50
)
pipe.attach(SlackSender)
pipe.attach(CHPrinter)

pipe.drain()
