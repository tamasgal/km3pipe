#!/usr/bin/env python
# Filename: ligiermirror.py
# Author: Tamas Gal <tgal@km3net.de>
# vim: ts=4 sw=4 et
"""
Subscribes to given tag(s) and sends them to another Ligier.

Usage:
    ligiermirror [options] SOURCE_IP
    ligiermirror (-h | --help)

Options:
    -t TARGET_IP    Target IP [default: 127.0.0.1].
    -p PORT         Source port [default: 5553].
    -q PORT         Target port [default: 5553].
    -m TAGS         Comma separated message tags [default: IO_EVT, IO_SUM].
    -s QUEUE        Maximum queue size for messages [default: 2000].
    -x TIMEOUT      Connection timeout in seconds [default: 604800].
    -d DEBUG_LEVEL  Debug level (DEBUG, INFO, WARNING, ...) [default: WARNING].
    -h --help       Show this screen.

"""
import socket

import km3pipe as kp


class LigierSender(kp.Module):
    """Forwards a message to another ligier"""

    def configure(self):
        target_ip = self.get("target_ip", default="127.0.0.1")
        port = self.get("port", default=5553)
        self.socket = socket.socket()
        self.client = self.socket.connect((target_ip, port))

    def process(self, blob):
        self.socket.send(blob["CHPrefix"].data + blob["CHData"])

    def finish(self):
        self.socket.close()


def main():
    """The main script"""
    from docopt import docopt
    args = docopt(__doc__, version=kp.version)

    kp.logger.set_level("km3pipe", args['-d'])

    pipe = kp.Pipeline()
    pipe.attach(
        kp.io.ch.CHPump,
        host=args['SOURCE_IP'],
        port=int(args['-p']),
        tags=args['-m'],
        timeout=int(args['-x']),
        max_queue=int(args['-s'])
    )
    pipe.attach(LigierSender, target_ip=args['-t'], port=int(args['-q']))
    pipe.drain()


if __name__ == '__main__':
    main()
