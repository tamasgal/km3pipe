#!/usr/bin/env python3
# Filename: daqsample.py
"""
Take samples from a given DAQ ControlHost stream.

Usage:
    daqsample [options] TAG OUTFILE
    daqsample (-h | --help)
    daqsample --version

Options:
    TAG             ControlHost TAG (e.g. IO_EVT).
    OUTFILE         Filename of the dump file.
    -i LIGIER_IP    IP of the Ligier [default: 127.0.0.1].
    -p LIGIER_PORT  Port of the Ligier [default: 5553].
    -n SAMPLES      Number of samples [default: 1].
    -h --help       Show this screen.

"""
import km3pipe as kp

__author__ = "Tamas Gal"
__copyright__ = "Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class Dumper(kp.Module):
    def configure(self):
        filename = self.require("filename")
        self._fobj = open(filename, "bw")

    def process(self, blob):
        self._fobj.write(blob["CHData"])
        return blob

    def finish(self):
        self._fobj.close()


def main():
    from docopt import docopt

    args = docopt(__doc__, version=kp.version)

    tag = args["TAG"]
    outfile = args["OUTFILE"]
    port = int(args["-p"])
    ip = args["-i"]
    n = int(args["-n"])

    pipe = kp.Pipeline()
    pipe.attach(kp.io.ch.CHPump, host=ip, port=port, tags=tag)
    pipe.attach(Dumper, filename=outfile)
    pipe.drain(n)
