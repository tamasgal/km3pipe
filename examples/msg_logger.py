#!/usr/bin/env python2.7
# coding=utf-8
# vim: ts=4 sw=4 et
# Author: Tamás Gál (tgal@km3net.de)
"""
JLigier MSG log dumper.

Usage:
    msg_logger.py [-i SECONDS] [-p PATH] DET_ID

Options:
    -h --help       Show this screen.
    -i SECONDS      Write interval in seconds [Default: 5].
    -p PATH         Target path to store the log files [Default: .].
    DET_ID          Detector ID (eg. 14).

"""
from __future__ import print_function

import threading
from time import sleep
import os
import urllib2
import json

import km3pipe as km3


RUN_NUMBER_URL='http://192.168.0.120:1301/mon/controlunit/runnumber'


class MessageDumper(km3.Module):
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.write_interval = self.get('write_interval') or 1
        self.path = self.get('path') or '.'
        self.det_id = self.get('det_id')
        self.messages = []

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.thread = None
        self.lock = threading.Lock()
        self._start_thread()

    @property
    def target_file(self):
        current_run = int(json.load(urllib2.urlopen(RUN_NUMBER_URL))['value'])
        return os.path.join(self.path, 'MSG_dump_{0:08}_{1:08}.log'
                                       .format(int(self.det_id), current_run))

    def process(self, blob):
        message = blob['CHData']
        with self.lock:
            self.messages.append(message)
        return blob

    def _start_thread(self):
        self.thread = threading.Thread(target=self._run, args=())
        self.thread.daemon = True
        self.thread.start()

    def _run(self):
        while True:
            self._dump(self.target_file)
            sleep(self.write_interval)

    def _dump(self, filename):
        with self.lock:
            print("Dumping {0} messages to {1}"
                  .format(len(self.messages), filename))
            with open(filename, 'a+') as f:
                for message in self.messages:
                    f.write(message + '\n')
            self.messages = []

    def finish(self):
        self._dump(self.target_file)


def main(det_id, target_path, write_interval):
    pipe = km3.Pipeline()
    pipe.attach(km3.pumps.CHPump, host='localhost',
                port=5553,
                tags='MSG',
                timeout=60*60*24*7,
                max_queue=2000)
    pipe.attach(MessageDumper,
                write_interval=write_interval, det_id=det_id, path=target_path)
    pipe.drain()


if __name__ == '__main__':
    from docopt import docopt
    arguments = docopt(__doc__, version=1.0)

    main(arguments['DET_ID'], arguments['-p'], int(arguments['-i']))
