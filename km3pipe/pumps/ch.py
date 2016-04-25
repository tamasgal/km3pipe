#!/usr/bin/env python
# coding=utf-8
# Filename: jpp.py
# pylint: disable=
"""
Pump for the jpp file read through aanet interface.

"""
from __future__ import division, absolute_import, print_function

from km3pipe import Pump
from km3pipe.logger import logging
import threading
import struct
import socket
from time import sleep
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty

log = logging.getLogger(__name__)  # pylint: disable=C0103


class CHPump(Pump):
    """A pump for ControlHost data."""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)

        self.host = self.get('host') or '127.0.0.1'
        self.port = self.get('port') or 5553
        self.tags = self.get('tags') or "MSG"
        self.timeout = self.get('timeout') or 60*60*24
        self.max_queue = self.get('max_queue') or 50
        self.key_for_data = self.get('key_for_data') or 'CHData'
        self.key_for_prefix = self.get('key_for_prefix') or 'CHPrefix'

        self.queue = Queue()
        self.client = None
        self.thread = None

        print("Connecting to {0} on port {1}\n"
              "Subscribed tags: {2}\n"
              "Connection timeout: {3}s\n"
              "Maximum queue size for incoming data: {4}"
              .format(self.host, self.port, self.tags, self.timeout,
                      self.max_queue))

        self._init_controlhost()
        self._start_thread()

    def _start_thread(self):
        self.thread = threading.Thread(target=self._run, args=())
        self.thread.daemon = True
        self.thread.start()

    def _init_controlhost(self):
        """Set up the controlhost connection"""
        from controlhost import Client
        self.client = Client(self.host, self.port)
        self.client._connect()
        for tag in self.tags.split(','):
            self.client.subscribe(tag.strip())

    def _run(self):
        while True:
            try:
                prefix, data = self.client.get_message()
            except struct.error:
                log.error("Corrupt data recieved, skipping...")
                continue
            if not data:
                log.critical("No data received, connection died.\n" +
                             "Trying to reconnect in 30 seconds.")
                sleep(30)
                try:
                    self._init_controlhost()
                except socket.error:
                    log.error("Failed to connect to host.")
                continue
            if self.queue.qsize() > self.max_queue:
                log.warn("Maximum queue size ({0}) reached, dropping data."
                         .format(self.max_queue))
            else:
                self.queue.put((prefix, data))

    def process(self, blob):
        """Wait for the next packet and put it in the blob"""
        try:
            prefix, data = self.queue.get(timeout=self.timeout)
        except Empty:
            log.warn("ControlHost timeout ({0}s) reached".format(self.timeout))
            raise StopIteration("ControlHost timeout reached.")
        blob[self.key_for_prefix] = prefix
        blob[self.key_for_data] = data
        return blob

    def finish(self):
        """Clean up the JLigier controlhost connection"""
        self.client._disconnect()
