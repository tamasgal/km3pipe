#!/usr/bin/env python
# coding=utf-8
# Filename: srv.py
from __future__ import print_function, absolute_import

"""
The KM3srv tornado webserver.

"""
__author__ = 'Tamas Gal'
__email__ = 'tgal@km3net.de'

import tornado.ioloop
import tornado.web
import tornado.websocket
from tornado.options import define, options

import json
import os
from thread import start_new_thread
import threading
import re
import subprocess
import math

from time import sleep

import pandas as pd

from km3pipe.core import Geometry
from km3pipe.logger import logging

log = logging.getLogger(__name__)

VERSION = '0.0.1'

define("ip", default="0.0.0.0", type=str,
       help="The WAN IP of this machine. You can use 127 for local tests.")
define("port", default="8088", type=int,
       help="The KM3srv server will be available on this port.")
define("data", default=os.path.expanduser("~/km3net/data"), type=str,
       help="Path to the data files.")

class EchoWebSocket(tornado.websocket.WebSocketHandler):
    """An echo handler for client/server messaging and debugging"""
    def __init__(self, *args, **kwargs):
        self.clients = kwargs.pop('clients')
        self.data_path = kwargs.pop('data_path')
        self._status = kwargs.pop('server_status')
        self._lock = kwargs.pop('lock')
        super(EchoWebSocket, self).__init__(*args, **kwargs)

    def check_origin(self, origin):
        return True

    def open(self):
        self.clients.append(self)
        welcome = u"WebSocket opened. Welcome to KM3srv {0}!".format(VERSION)
        print(welcome)
        self.message(welcome)
        self.message(self.status, 'status')

    def on_close(self):
        self.clients.remove(self)
        print("WebSocket closed, client removed.")

    def on_message(self, message):
        # self.message(u"Client said '{0}'".format(message))
        print("Client said: {0}".format(message))
        if message.startswith('event'):
            p = re.compile(ur'event/(\d+)/(\d+)/(\d+)')
            try:
                det_id, run_id, event_id = re.search(p, message).groups()
            except AttributeError:
                self.message("Syntax error, try event/DET_ID/RUN_ID/EVENT")
            else:
                start_new_thread(self.get_event,
                                 (int(det_id), int(run_id), int(event_id)))

    def get_event(self, det_id, run_id, event_id):
        det_dir_name = "KM3NeT_{0:08d}".format(det_id)
        det_dir = os.path.join(self.data_path, 'sea', det_dir_name)
        sub_dir = str(int(math.floor(run_id / 1000)))
        data_dir = os.path.join(det_dir, sub_dir)

        basename = "{0}_{1:08d}".format(det_dir_name, run_id, event_id)
        h5filename = basename + ".h5"
        rootfilename = basename + ".root"

        irods_path = os.path.join("/in2p3/km3net/data/raw/sea",
                                  det_dir_name, sub_dir, rootfilename)

        h5filepath = os.path.join(data_dir, h5filename)
        h5filepath_tmp = h5filepath + '.tmp'
        rootfilepath = os.path.join(data_dir, rootfilename)

        self.message("Looking for {0}".format(basename))
        print("Request for event {0} in {1}".format(event_id, h5filename))
        print("Detector dir: {0}".format(det_dir))
        print("Data dir: {0}".format(data_dir))

        if os.path.exists(h5filepath_tmp):
            self.message("File is currently being process. Waiting...")

        while os.path.exists(h5filepath_tmp):
            sleep(3)

        if not os.path.exists(h5filepath):
            self.status = 'busy'
            print("Creating {0}".format(h5filepath_tmp))
            subprocess.call(['mkdir', '-p', data_dir])
            subprocess.call(['touch', h5filepath_tmp])
            if not os.path.exists(rootfilepath):
                self.message("No HDF5 file found, downloading ROOT file.")
                print("Retrieve {0}".format(irods_path))
                status = subprocess.call(['iget', irods_path, data_dir])
                if status == 0:
                    self.message("Successfully downloaded data.")
                else:
                    self.message("There was a problem downloading the data.")
                    return
            status = subprocess.call(['km3pipe', 'jpptohdf5',
                                      '-i', rootfilepath,
                                      '-o', h5filepath_tmp])
            if status == 0:
                self.message("Successfully converted data.")
                os.rename(h5filepath_tmp, h5filepath)
            else:
                self.message("There was a problem converting the data.")

        if os.path.exists(rootfilepath):
            log.warn("Removing ROOT file {0}".format(rootfilepath))
            os.remove(rootfilepath)

        hits = pd.read_hdf(h5filepath, 'hits')
        snapshot_hits = hits[(hits['event_id'] == event_id)].copy()
        triggered_hits = hits[(hits['event_id'] == event_id) &
                              (hits['triggered'] == True)]
        self.message("Det ID: {0} Run ID: {1} Event ID: {2} - "
                     "Snapshot hits: {3}, Triggered hits: {4}"
                     .format(det_id, run_id, event_id,
                             len(snapshot_hits), len(triggered_hits)))
        geometry = Geometry(det_id=det_id)
        geometry.apply(snapshot_hits)

        self.message(snapshot_hits.to_dict(), "event")
        self.status = 'ready'

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = value
        self.broadcast(self.status, kind='status')

    def message(self, data, kind="info"):
        """Convert message to json and send it to the clients"""
        #message = json.dumps({'kind': kind, 'data': data})
        message = pd.io.json.dumps({'kind': kind, 'data': data})
        print("Sent {0} bytes.".format(len(message)))
        self.write_message(message)

    def broadcast(self, text, kind="info"):
        log.info("Number of connected clients: {0}".format(len(self.clients)))
        for client in self.clients:
            message = json.dumps({'kind': kind, 'data': text})
            try:
                client.write_message(message)
            except tornado.websocket.WebSocketClosedError:
                log.error("Lost connection to client '{0'}".format(client))


def main():
    root = os.path.dirname(__file__)
    cwd = os.getcwd()

    options.parse_command_line()

    ip = options.ip
    port = int(options.port)
    data_path = options.data
    server_status = 'ready'
    lock = threading.Lock()
    clients = []

    print("Starting KM3srv.")
    print("Running on {0}:{1}".format(ip, port))
    print("Data path: {0}".format(data_path))

    settings = {'debug': True,
                'static_path': os.path.join(root, 'static'),
                'template_path': os.path.join(root, 'static/templates'),
                }


    application = tornado.web.Application([
        (r"/test", EchoWebSocket, {'clients': clients,
                                   'server_status': server_status,
                                   'data_path': data_path,
                                   'lock': lock}),
    ], **settings)

    try:
        application.listen(port)
        tornado.ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        print("Stopping tornado...")
        tornado.ioloop.IOLoop.instance().stop()


if __name__ == "__main__":
    main()


