#!/usr/bin/env python
# coding=utf-8
# Filename: jpp.py
# pylint: disable=
"""
Pump for the jpp file read through aanet interface.

"""
from __future__ import division, absolute_import, print_function

from km3pipe import Pump
from km3pipe.tools import Cuckoo
from km3pipe.logger import logging
import threading
import struct
import socket
from time import sleep
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

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
        self.cuckoo_warn = Cuckoo(60*5, log.warn)

        self.loop_cycle = 0
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
        log.debug("Starting and demonising thread.")
        self.thread = threading.Thread(target=self._run, args=())
        self.thread.daemon = True
        self.thread.start()

    def _init_controlhost(self):
        """Set up the controlhost connection"""
        log.debug("Connecting to JLigier")
        from controlhost import Client
        self.client = Client(self.host, self.port)
        self.client._connect()
        log.debug("Subscribing to tags: {0}".format(self.tags))
        for tag in self.tags.split(','):
            self.client.subscribe(tag.strip())
        log.debug("Controlhost initialisation done.")

    def _run(self):
        log.debug("Entering the main loop.")
        while True:
            current_qsize = self.queue.qsize()
            log.info("----- New loop cycle #{0}".format(self.loop_cycle))
            log.info("Current queue size: {0}".format(current_qsize))
            self.loop_cycle += 1
            try:
                log.debug("Waiting for data from network...")
                prefix, data = self.client.get_message()
                log.debug("{0} bytes received from network.".format(len(data)))
            except struct.error:
                log.error("Corrupt data recieved, skipping...")
                continue
            if not data:
                log.critical("No data received, connection died.\n" +
                             "Trying to reconnect in 30 seconds.")
                sleep(30)
                try:
                    log.debug("Reinitialising new CH connection.")
                    self._init_controlhost()
                except socket.error:
                    log.error("Failed to connect to host.")
                continue
            if current_qsize > self.max_queue:
                self.cuckoo_warn("Maximum queue size ({0}) reached, "
                                 "dropping data.".format(self.max_queue))
            else:
                log.debug("Filling data into queue.")
                self.queue.put((prefix, data))
        log.debug("Quitting the main loop.")

    def process(self, blob):
        """Wait for the next packet and put it in the blob"""
        try:
            log.debug("Waiting for queue items.")
            prefix, data = self.queue.get(timeout=self.timeout)
            log.debug("Got {0} bytes from queue.".format(len(data)))
        except Empty:
            log.warn("ControlHost timeout ({0}s) reached".format(self.timeout))
            raise StopIteration("ControlHost timeout reached.")
        blob[self.key_for_prefix] = prefix
        blob[self.key_for_data] = data
        return blob

    def finish(self):
        """Clean up the JLigier controlhost connection"""
        log.debug("Disconnecting from JLigier.")
        self.client._disconnect()
