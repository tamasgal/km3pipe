#!/usr/bin/env python
# Filename: jpp.py
# pylint: disable=
"""
Pump for the jpp file read through aanet interface.

"""

from thepipe import Module, Blob
from km3pipe.controlhost import Client
from km3pipe.time import Cuckoo
from km3pipe.logger import get_logger
import threading
import socket
import time
import numpy as np
from collections import deque
from queue import Queue, Empty

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = get_logger(__name__)  # pylint: disable=C0103


class CHPump(Module):
    """A pump for ControlHost data."""

    def configure(self):
        self.host = self.get("host") or "127.0.0.1"
        self.port = self.get("port") or 5553
        self.tags = self.get("tags") or "MSG"
        self.timeout = self.get("timeout") or 60 * 60 * 24
        self.max_queue = self.get("max_queue") or 50
        self.key_for_data = self.get("key_for_data") or "CHData"
        self.key_for_prefix = self.get("key_for_prefix") or "CHPrefix"
        self.subscription_mode = self.get("subscription_mode", default="wait")
        self.show_statistics = self.get("show_statistics", default=False)
        self.statistics_interval = self.get("statistics_interval", default=30)
        self.cuckoo_warn = Cuckoo(60 * 5, log.warning)
        self.performance_warn = Cuckoo(
            self.statistics_interval, self.show_performance_statistics
        )

        self.idle_dt = deque(maxlen=1000)
        self.idle_timer = time.time()
        self.message_count = 0

        self.loop_cycle = 0
        self.queue = Queue()
        self.client = None
        self.thread = None

        if self.subscription_mode == "all":
            self.log.warning(
                "You subscribed to the ligier in 'all'-mode! "
                "If you are too slow with data processing, "
                "you will block other clients. "
                "If you don't understand this message "
                "and are running this code on a DAQ machine, "
                "consult a DAQ expert now and stop this script."
            )

        print(
            "Connecting to {0} on port {1}\n"
            "Subscribed tags: {2}\n"
            "Connection timeout: {3}s\n"
            "Maximum queue size for incoming data: {4}".format(
                self.host, self.port, self.tags, self.timeout, self.max_queue
            )
        )

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
        self.client = Client(self.host, self.port)
        self.client._connect()
        log.debug("Subscribing to tags: {0}".format(self.tags))
        for tag in self.tags.split(","):
            self.client.subscribe(tag.strip(), mode=self.subscription_mode)
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
                self.message_count += 1
                self._add_idle_dt()
                self._set_idle_timer()
                self.performance_warn()
                log.debug("{0} bytes received from network.".format(len(data)))
            except EOFError:
                log.warning("EOF from Ligier, trying again in 30 seconds...")
                time.sleep(30)
                continue
            except BufferError:
                log.error("Buffer error in Ligier stream, aborting...")
                break
            if not data:
                log.critical(
                    "No data received, connection died.\n"
                    + "Trying to reconnect in 30 seconds."
                )
                time.sleep(30)
                try:
                    log.debug("Reinitialising new CH connection.")
                    self._init_controlhost()
                except socket.error:
                    log.error("Failed to connect to host.")
                continue
            if current_qsize > self.max_queue:
                self.cuckoo_warn(
                    "Maximum queue size ({0}) reached, "
                    "dropping data.".format(self.max_queue)
                )
            else:
                log.debug("Filling data into queue.")
                self.queue.put((prefix, data))
            self._set_idle_timer()
        log.debug("Quitting the main loop.")

    def process(self, blob):
        """Wait for the next packet and put it in the blob"""
        try:
            log.debug("Waiting for queue items.")
            prefix, data = self.queue.get(timeout=self.timeout)
            log.debug("Got {0} bytes from queue.".format(len(data)))
        except Empty:
            log.warning("ControlHost timeout ({0}s) reached".format(self.timeout))
            raise StopIteration("ControlHost timeout reached.")
        blob[self.key_for_prefix] = prefix
        blob[self.key_for_data] = data
        return blob

    def show_performance_statistics(self):
        if not self.show_statistics:
            return
        dt = np.median(self.idle_dt)
        current_qsize = self.queue.qsize()
        log_func = self.cprint
        if dt < 0 or current_qsize > 0:
            log_func = self.log.warning
        log_func(
            "Message rate: {0:.1f} Hz, median idle time per message: "
            "{1:.3f} us (current queue size: {2})".format(
                self.message_count / self.statistics_interval, dt * 1e6, current_qsize
            )
        )
        self.message_count = 0

    def _set_idle_timer(self):
        self.idle_timer = time.time()

    def _add_idle_dt(self):
        now = time.time()
        self.idle_dt.append(now - self.idle_timer)

    def finish(self):
        """Clean up the JLigier controlhost connection"""
        log.debug("Disconnecting from JLigier.")
        self.client.socket.shutdown(socket.SHUT_RDWR)
        self.client._disconnect()

    def __iter__(self):
        return self

    def __next__(self):
        return self.process(Blob())

    def next(self):
        return self.__next__()


def CHTagger(blob):
    tag = str(blob["CHPrefix"].tag)
    blob[tag] = True
    return blob
