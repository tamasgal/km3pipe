#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: ts=4 sw=4 et
"""
=============
Ligier Mirror
=============

Subscribes to given tag(s) and sends them to another Ligier.
This script is also available as a command line utility in km3pipe, which can
be accessed by the command ``ligiermirror``.

"""
from __future__ import absolute_import, print_function, division

# Author: Tamas Gal <tgal@km3net.de>
# License: MIT

import socket

from km3pipe import Pipeline, Module
from km3pipe.io import CHPump


class LigierSender(Module):
    def configure(self):
        self.ligier = self.get("ligier") or "127.0.0.1"
        self.port = self.get("port") or 5553
        self.socket = socket.socket()
        self.client = self.socket.connect((self.ligier, self.port))

    def process(self, blob):
        self.socket.send(blob["CHPrefix"].data + blob["CHData"])

    def finish(self):
        self.socket.close()


pipe = Pipeline()
pipe.attach(
    CHPump,
    host='192.168.0.121',
    port=5553,
    tags='IO_EVT, IO_SUM, IO_TSL',
    timeout=60 * 60 * 24 * 7,
    max_queue=2000
)
pipe.attach(LigierSender)
pipe.drain()
