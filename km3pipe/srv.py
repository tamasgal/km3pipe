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
import thread

from time import sleep

VERSION = '0.0.1'

define("ip", default="0.0.0.0", type=str,
       help="The WAN IP of this machine. You can use 127 for local tests.")
define("port", default="8088", type=int,
       help="The KM3srv server will be available on this port.")

class EchoWebSocket(tornado.websocket.WebSocketHandler):
    """An echo handler for client/server messaging and debugging"""
    def __init__(self, *args, **kwargs):
        self.clients = kwargs.pop('clients')
        super(EchoWebSocket, self).__init__(*args, **kwargs)

    def check_origin(self, origin):
        return True

    def open(self):
        welcome = u"WebSocket opened. Welcome to KM3srv {0}!".format(VERSION)
        print(welcome)
        self.send_json_message(welcome)
        self.clients.append(self)

    def on_message(self, message):
        self.send_json_message(u"Client said '{0}'".format(message))
        thread.start_new_thread(self.background_process, (message, ))

    def background_process(self, foo):
        self.write_message("Grabbing {0}".format(foo))
        self.write_message("Sleeping for 7 seconds for {0}...".format(foo))
        sleep(7)
        self.write_message("Waking up in 12 seconds for {0}...".format(foo))
        sleep(12)
        self.write_message("Done with {0}!".format(foo))

    def send_json_message(self, text):
        """Convert message to json and send it to the clients"""
        message = json.dumps({'kind': 'message', 'text': text})
        self.write_message(message)


def main():
    root = os.path.dirname(__file__)
    cwd = os.getcwd()

    options.parse_command_line()

    ip = options.ip
    port = int(options.port)

    print("Starting KM3srv.")
    print("Running on {0}:{1}".format(ip, port))

    settings = {'debug': True,
                'static_path': os.path.join(root, 'static'),
                'template_path': os.path.join(root, 'static/templates'),
                }

    clients = []

    application = tornado.web.Application([
        (r"/test", EchoWebSocket, {'clients': clients}),
    ], **settings)

    try:
        application.listen(port)
        tornado.ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        print("Stopping tornado...")
        tornado.ioloop.IOLoop.instance().stop()


if __name__ == "__main__":
    main()


