#!/usr/bin/env python
# coding=utf-8
# Filename: online_reco.py
# Author: Tamas Gal <tgal@km3net.de>
# vim: ts=4 sw=4 et
"""
Visualisation routines for online reconstruction.

Usage:
    online_reco.py [options]
    online_reco.py (-h | --help)

Options:
    -l LIGIER_IP    The IP of the ligier [default: 127.0.0.1].
    -p LIGIER_PORT  The port of the ligier [default: 5553].
    -o PLOT_DIR     The directory to save the plot [default: www/plots].
    -h --help       Show this screen.

"""
from collections import deque
import km3pipe as kp


class ZenithDistribution(kp.Module):
    def configure(self):
        self.max_events = 1000
        self.zeniths = deque(maxlen=1000)

    def process(self, blob):
        print(blob.keys())
        print(blob['RecoTrack'])
        return blob


def main():
    from docopt import docopt
    args = docopt(__doc__)

    plots_path = args['-o']
    ligier_ip = args['-l']
    ligier_port = int(args['-p'])

    pipe = kp.Pipeline()
    pipe.attach(
        kp.io.ch.CHPump,
        host=ligier_ip,
        port=ligier_port,
        tags='IO_OLINE',
        timeout=60 * 60 * 24 * 7,
        max_queue=2000
    )
    pipe.attach(kp.io.daq.DAQProcessor)
    pipe.attach(ZenithDistribution, plots_path=plots_path)
    pipe.drain()


if __name__ == '__main__':
    main()
