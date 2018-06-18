#!/usr/bin/env python
# vim: ts=4 sw=4 et
# Author: Tamás Gál (tgal@km3net.de)
"""
JLigier MSG log dumper.

Usage:
    msg_logger.py [-i SECONDS] [-p PATH] DET_SN

Options:
    -h --help       Show this screen.
    -i SECONDS      Write interval in seconds [Default: 5].
    -p PATH         Target path to store the log files [Default: .].
    DET_SN          Detector serial number (eg. 14).

"""

import threading
from time import sleep
import os
import urllib2
import json

import km3pipe as km3
from km3pipe.logger import get_logger

from pyslack import SlackClient

__author__ = 'tamasgal'

log = get_logger(__name__)
RUN_NUMBER_URL = 'http://192.168.0.120:1301/mon/controlunit/runnumber'


class MessageDumper(km3.Module):
    def configure(self):
        self.write_interval = self.get('write_interval') or 1
        self.path = self.get('path') or '.'
        self.det_sn = self.get('det_sn')
        self.messages = []
        self.slack = SlackClient(km3.config.Config().slack_token)
        self.cuckoo = km3.tools.Cuckoo(interval=5, callback=self.send_message)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.thread = None
        self.lock = threading.Lock()
        self._start_thread()

    @property
    def target_file(self):
        current_run = int(json.load(urllib2.urlopen(RUN_NUMBER_URL))['value'])
        return os.path.join(
            self.path,
            'MSG_dump_{0:08}_{1:08}.log'.format(int(self.det_sn), current_run)
        )

    def process(self, blob):
        message = blob['CHData']
        if 'ERROR' in message:
            self.cuckoo.msg(message)
            log.error(message)
        with self.lock:
            self.messages.append(message)
        return blob

    def send_message(self, message):
        self.slack.chat_post_message(
            "#live-arca-it", message, username="ligier"
        )

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
            print(
                "Dumping {0} messages to {1}".format(
                    len(self.messages), filename
                )
            )
            with open(filename, 'a+') as f:
                for message in self.messages:
                    f.write(message + '\n')
            self.messages = []

    def finish(self):
        self._dump(self.target_file)


def main(det_sn, target_path, write_interval):
    pipe = km3.Pipeline()
    pipe.attach(
        km3.io.CHPump,
        host='localhost',
        port=5553,
        tags='MSG',
        timeout=60 * 60 * 24 * 7,
        max_queue=2000
    )
    pipe.attach(
        MessageDumper,
        write_interval=write_interval,
        det_sn=det_sn,
        path=target_path
    )
    pipe.drain()


if __name__ == '__main__':
    from docopt import docopt
    arguments = docopt(__doc__, version=1.0)

    main(arguments['DET_SN'], arguments['-p'], int(arguments['-i']))
