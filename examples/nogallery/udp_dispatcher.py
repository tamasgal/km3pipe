#!/usr/bin/env python
# coding=utf-8
# vim: ts=4 sw=4 et
from __future__ import print_function

import socket
import sys

import km3pipe as kp

__author__ = "Tamas Gal"
__email__ = "tgal@km3net.de"


class UDPForwarder(kp.Module):
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def process(self, blob):
        if str(blob['CHPrefix'].tag) == 'IO_MONIT':
            self.sock.sendto(blob['CHData'], ('127.0.0.1', 50000))
            #sys.stdout.write('.')
            #sys.stdout.flush()
        return blob



pipe = kp.Pipeline()
pipe.attach(kp.io.CHPump,
            host='localhost',
            port=31286,
            tags='IO_MONIT',
            timeout=60*60*24*7,
            max_queue=1000)
pipe.attach(UDPForwarder)
pipe.drain()
