#!/usr/bin/env python
from __future__ import division
import time
from time import ctime
from pyslack import SlackClient

from km3pipe import Pipeline, Module
from km3pipe.io import CHPump
from km3pipe.io.daq import DAQPreamble, DAQEvent
from km3pipe.common import StringIO


class CHPrinter(Module):
    def process(self, blob):
        print("New blob:")
        print blob['CHPrefix']
        return blob


class SlackSender(Module):
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.client = SlackClient('YOUR_SLACK_API_TOKEN_HERE')
        self.current_run = None

    def process(self, blob):
        data_size = blob['CHPrefix'].length

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
            self.client.chat_post_message("#du2-live",
                                          "Run #{0} has started!".format(event.header.run),
                                          username="slackbot")

        return blob


pipe = Pipeline()
pipe.attach(CHPump, host='localhost', # You'll need a VPN connection and several SSH tunnels
                    port=5553,
                    tags='IO_EVT',
                    timeout=60*60*24,
                    max_queue=50)
pipe.attach(SlackSender)
pipe.attach(CHPrinter)

pipe.drain()
