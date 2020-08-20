#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: ts=4 sw=4 et
"""
=============
Log.io Module
=============

A module to send live log messages to a log.io server.
"""

# Author: Tamas Gal <tgal@km3net.de>
# License: MIT
import socket

from km3pipe import Pipeline, Module
from km3pipe.pumps import CHPump


class LogIO(Module):
    def configure(self):
        url = "pi2089.physik.uni-erlangen.de"
        port = 28777

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((url, port))

    def process(self, blob):
        data = blob["CHData"]
        log_level = "info"
        if "ERROR" in data:
            log_level = "error"
        if "WARNING" in data:
            log_level = "warning"
        source = "Other"
        if " F0" in data:
            source = "DataFilter"
        if " Q0" in data:
            source = "DataQueue"
        if " W0" in data:
            source = "DataWriter"
        self.sock.send(
            "+log|{0}|Portopalo DAQ|{1}|{2}\r\n".format(source, log_level, data)
        )
        return blob

    def finish(self):
        self.sock.close()


pipe = Pipeline()
pipe.attach(
    CHPump, host="127.0.0.1", port=5553, tags="MSG", timeout=60 * 60 * 24, max_queue=500
)
pipe.attach(LogIO)
pipe.drain()
