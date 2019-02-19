#!/usr/bin/env python
# Filename: srv.py
"""
The KM3srv tornado webserver.

"""
from __future__ import absolute_import, print_function, division

try:
    import tornado
except ImportError:
    print(
        "Please 'pip install tornado websocket-client' "
        "to use the km3srv package"
    )
    exit(1)

import tornado.ioloop
import tornado.web
import tornado.websocket
from tornado.options import define, options

import os
import threading
import re
import subprocess
import math

from time import sleep

import pandas as pd
import websocket

from .calib import Calibration
from .config import Config
from .dataclasses import Table
from .tools import token_urlsafe
from .logger import get_logger

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = get_logger(__name__)

VERSION = '0.0.1'

define(
    "ip",
    default="0.0.0.0",
    type=str,
    help="The WAN IP of this machine. You can use 127 for local tests."
)
define(
    "port",
    default="8088",
    type=int,
    help="The KM3srv server will be available on this port."
)
define(
    "data",
    default=os.path.expanduser("~/km3net/data"),
    type=str,
    help="Path to the data files."
)

RBA_URL = Config().rba_url


class ClientManager(object):
    """Manage km3srv clients.
    """

    def __init__(self):
        self._clients = {}
        threading.Thread(target=self.heartbeat).start()

    def add(self, client):
        token = token_urlsafe(3)
        self._clients[token] = client
        log.info("New client with token '{0}' registered.".format(token))
        return token

    def remove(self, token):
        del self._clients[token]
        print("Client with token '{0}' removed.".format(token))

    def heartbeat(self, interval=30):
        while True:
            log.info("Pinging {0} clients.".format(len(self._clients)))
            print(self._clients)
            for client in self._clients.values():
                print(client)
                client.message("Ping.")
            sleep(interval)

    def broadcast_status(self):
        # status = {
        #    n_clients : len(self._clients)
        # }
        self.broadcast(
            "Number of connected clients: {0}.".format(len(self._clients))
        )

    def message_to(self, token, data, kind):
        message = pd.io.json.dumps({'kind': kind, 'data': data})
        print("Sent {0} bytes.".format(len(message)))
        self.raw_message_to(token, message)

    def raw_message_to(self, token, message):
        """Convert message to JSON and send it to the client with token"""
        if token not in self._clients:
            log.critical("Client with token '{0}' not found!".format(token))
            return
        client = self._clients[token]
        try:
            client.write_message(message)
        except (AttributeError, tornado.websocket.WebSocketClosedError):
            log.error("Lost connection to client '{0}'".format(client))
        else:
            print("Sent {0} bytes.".format(len(message)))

    def broadcast(self, data, kind="info"):
        log.info("Broatcasting to {0} clients.".format(len(self._clients)))
        for token in self._clients.keys():
            self.message_to(token, data, kind)


class MessageProvider(tornado.websocket.WebSocketHandler):
    def __init__(self, *args, **kwargs):
        self.client_manager = kwargs.pop('client_manager')
        super(MessageProvider, self).__init__(*args, **kwargs)

    def on_message(self, message):
        log.warning("Client said: {0}".format(message))
        try:
            token = pd.io.json.loads(message)['token']
        except (ValueError, KeyError):
            log.error("Invalid JSON received: {0}".format(message))
        else:
            self.client_manager.raw_message_to(token, message)


class EchoWebSocket(tornado.websocket.WebSocketHandler):
    """An echo handler for client/server messaging and debugging"""

    def __init__(self, *args, **kwargs):
        log.warning("EchoWebSocket created!")
        self.client_manager = kwargs.pop('client_manager')
        self.data_path = kwargs.pop('data_path')
        self._status = kwargs.pop('server_status')
        self._lock = kwargs.pop('lock')
        self._token = self.client_manager.add(self)
        super(EchoWebSocket, self).__init__(*args, **kwargs)

    def check_origin(self, origin):
        return True

    def open(self):
        welcome = u"WebSocket opened. Welcome to KM3srv {0}!".format(VERSION)
        self.message(welcome)
        self.message(self._token, 'token')
        self.message(self.status, 'status')
        self.client_manager.broadcast_status()

    def on_close(self):
        self.client_manager.remove(self._token)
        print("WebSocket closed, client removed.")

    def on_message(self, message):
        # self.message(u"Client said '{0}'".format(message))
        print("Client said: {0}".format(message))
        if message.startswith('event'):
            p = re.compile(r'event/(\d+)/(\d+)/(\d+)')
            try:
                det_id, run_id, event_id = re.search(p, message).groups()
            except AttributeError:
                self.message("Syntax error, try event/DET_ID/RUN_ID/EVENT")
            else:
                threading.Thread(
                    target=self.get_event,
                    args=(int(det_id), int(run_id), int(event_id))
                ).start()

    def get_event(self, det_id, run_id, event_id):
        det_dir_name = "KM3NeT_{0:08d}".format(det_id)
        det_dir = os.path.join(self.data_path, 'sea', det_dir_name)
        sub_dir = str(int(math.floor(run_id / 1000)))
        data_dir = os.path.join(det_dir, sub_dir)

        basename = "{0}_{1:08d}".format(det_dir_name, run_id, event_id)
        h5filename = basename + ".h5"
        rootfilename = basename + ".root"

        irods_path = os.path.join(
            "/in2p3/km3net/data/raw/sea", det_dir_name, sub_dir, rootfilename
        )

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
            status = subprocess.call([
                'km3pipe', 'jpptohdf5', '-i', rootfilepath, '-o',
                h5filepath_tmp
            ])
            if status == 0:
                self.message("Successfully converted data.")
                os.rename(h5filepath_tmp, h5filepath)
            else:
                self.message("There was a problem converting the data.")

        if os.path.exists(rootfilepath):
            log.warning("Removing ROOT file {0}".format(rootfilepath))
            os.remove(rootfilepath)

        hits = pd.read_hdf(h5filepath, 'hits', mode='r')
        snapshot_hits = hits[(hits['event_id'] == event_id)].copy()
        triggered_hits = hits[(hits['event_id'] == event_id) &
                              (hits['triggered'] == True)]    # noqa
        self.message(
            "Det ID: {0} Run ID: {1} Event ID: {2} - "
            "Snapshot hits: {3}, Triggered hits: {4}".format(
                det_id, run_id, event_id, len(snapshot_hits),
                len(triggered_hits)
            )
        )
        cal = Calibration(det_id=det_id)
        cal.apply(snapshot_hits)

        event = {
            "hits": {
                'pos': [
                    tuple(x) for x in snapshot_hits[['x', 'y', 'z']].values
                ],
                'time': list(snapshot_hits['time']),
                'tot': list(snapshot_hits['tot']),
            }
        }

        self.message(event, "event")
        self.status = 'ready'

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = value
        # self.broadcast(self.status, kind='status')

    def message(self, data, kind="info"):
        """Convert message to json and send it to the clients"""
        message = pd.io.json.dumps({'kind': kind, 'data': data})
        print("Sent {0} bytes.".format(len(message)))
        self.write_message(message)


def srv_event(token, hits, url=RBA_URL):
    """Serve event to RainbowAlga"""

    if url is None:
        log.error("Please provide a valid RainbowAlga URL.")
        return

    ws_url = url + '/message'

    if isinstance(hits, pd.core.frame.DataFrame):
        pos = [tuple(x) for x in hits[['x', 'y', 'z']].values]
        time = list(hits['time'])
        tot = list(hits['tot'])
    elif isinstance(hits, Table):
        pos = list(zip(hits.pos_x, hits.pos_y, hits.pos_z))
        time = list(hits.time)
        tot = list(hits.tot)
    else:
        log.error(
            "No calibration information found in hits (type: {0})".format(
                type(hits)
            )
        )
        return

    event = {
        "hits": {
            'pos': pos,
            'time': time,
            'tot': tot,
        }
    }

    srv_data(ws_url, token, event, 'event')


def srv_data(url, token, data, kind):
    """Serve data to RainbowAlga"""
    ws = websocket.create_connection(url)
    message = {'token': token, 'data': data, 'kind': kind}
    ws.send(pd.io.json.dumps(message))
    ws.close()


def main():
    root = os.path.dirname(__file__)

    options.parse_command_line()

    ip = options.ip
    port = int(options.port)
    data_path = options.data
    server_status = 'ready'
    lock = threading.Lock()
    client_manager = ClientManager()

    print("Starting KM3srv.")
    print("Running on {0}:{1}".format(ip, port))
    print("Data path: {0}".format(data_path))

    settings = {
        'debug': True,
        'static_path': os.path.join(root, 'static'),
        'template_path': os.path.join(root, 'static/templates'),
    }

    application = tornado.web.Application([
        (
            r"/test", EchoWebSocket, {
                'client_manager': client_manager,
                'server_status': server_status,
                'data_path': data_path,
                'lock': lock
            }
        ),
        (r"/message", MessageProvider, {
            'client_manager': client_manager
        }),
    ], **settings)

    try:
        application.listen(port)
        tornado.ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        print("Stopping tornado...")
        tornado.ioloop.IOLoop.instance().stop()


if __name__ == "__main__":
    main()
