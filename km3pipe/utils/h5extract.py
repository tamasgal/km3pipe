#!/usr/bin/env python3
# Filename: h5extract.py
"""
A tool to extract data from KM3NeT ROOT files to HDF5.

Usage:
    h5extract [-o OUTFILE] FILENAME
    h5extract (-h | --help)
    h5extract --version

Options:
    -o OUTFILE          Output file.
    -h --help           Show this screen.
    --version           Show the version.

"""
import km3pipe as kp
import km3modules as km

log = kp.logger.get_logger(__name__)


def blob_printer(blob):
    print(blob)


def main():
    from docopt import docopt

    args = docopt(__doc__, version=kp.version)

    print(args)

    outfile = args['-o']
    if outfile is None:
        outfile = args['FILENAME'] + ".h5"

    pipe = kp.Pipeline(timeit=True)
    pipe.attach(kp.io.OfflinePump, filename=args['FILENAME'])
    pipe.attach(km.StatusBar, every=1)
    pipe.attach(km.io.HitsTabulator)
    pipe.attach(blob_printer)
    pipe.attach(kp.io.HDF5Sink, filename=outfile)
    pipe.drain()
