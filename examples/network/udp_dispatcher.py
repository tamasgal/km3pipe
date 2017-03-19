#!/usr/bin/env python
# coding=utf-8
# vim: ts=4 sw=4 et
"""
============================
UDP Forwarder for ControlHost
============================

A simple UDP forwarder for ControlHost messages.
"""
# Author: Tamas Gal <tgal@km3net.de>
# License: MIT
#!/usr/bin/env python
# coding=utf-8
# vim: ts=4 sw=4 et
"""
This application is used to forward monitoring channel data from Ligier
to a given UDP address.

"""
from __future__ import print_function

import socket
import sys

import km3pipe as kp

__author__ = "Tamas Gal"
__email__ = "tgal@km3net.de"


class UDPForwarder(kp.Module):
    def configure(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.counter = 0

    def process(self, blob):
        if str(blob['CHPrefix'].tag) == 'IO_MONIT':
            self.sock.sendto(blob['CHData'], ('127.0.0.1', 56017))
            if self.counter % 100 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            self.counter += 1
        return blob



pipe = kp.Pipeline()
pipe.attach(kp.io.CHPump,
            host='localhost',
            port=5553,
            tags='IO_MONIT',
            timeout=60*60*24*7,
            max_queue=1000,
            timeit=True)
pipe.attach(UDPForwarder)
pipe.drain()
