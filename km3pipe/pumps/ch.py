#!/usr/bin/env python
# coding=utf-8
# Filename: jpp.py
# pylint: disable=
"""
Pump for the jpp file read through aanet interface.

"""
from __future__ import division, absolute_import, print_function

import socket

from km3pipe import Pump
from km3pipe.logger import logging


log = logging.getLogger(__name__)  # pylint: disable=C0103


class CHPump(Pump):
    """A pump for ControlHost data."""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)

        self.host = self.get('host') or '127.0.0.1'
        self.port = self.get('port') or 5553
        self.tag = self.get('tag') or 'MSG'
        self.key_for_data = self.get('key_for_data') or 'CHData'
        self.key_for_prefix = self.get('key_for_prefix') or 'CHPrefix'

        self.client = None

        self._init_controlhost()

    def _init_controlhost(self):
        """Set up the controlhost connection"""
        from controlhost import Client
        self.client = Client(self.host, self.port)
        self.client._connect()
        self.client.subscribe(self.tag)

    def process(self, blob):
        """Wait for the next packet and put it in the blob"""
        try:
            prefix, data = self.client.get_message()
        except socket.error:
            log.warn("Stopping cycle due to socket error")
            raise StopIteration("Stopping cycle due to socket error")
        else:
            blob[self.key_for_prefix] = prefix
            blob[self.key_for_data] = data
            return blob
        
    def finish(self):
        """Clean up the JLigier controlhost connection"""
        self.ch_client._disconnect()
