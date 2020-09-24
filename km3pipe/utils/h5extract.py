#!/usr/bin/env python3
# Filename: h5extract.py
"""
A tool to extract data from KM3NeT ROOT files to HDF5.

Usage:
    h5extract [options] FILENAME
    h5extract (-h | --help)
    h5extract --version

Options:
    -o OUTFILE            Output file.
    --offline-header      Extract the offline header.
    --event-info          Extract event information.
    --offline-hits        Extract offline hits.
    --mc-hits             Extract MC hits.
    --online-hits         Extract snapshot and triggered hits (combined).
    --mc-tracks           Extract MC tracks.
    --mc-tracks-usr-data  Extract usr data from MC tracks (this will be slow).
    --timeit              Print detailed pipeline performance statistics.
    -h --help             Show this screen.
    --version             Show the version.

"""
import km3pipe as kp
import km3modules as km

log = kp.logger.get_logger(__name__)


def blob_printer(blob):
    print(blob)
    return blob


def main():
    from docopt import docopt

    args = docopt(__doc__, version=kp.version)

    # print(args)

    outfile = args["-o"]
    if outfile is None:
        outfile = args["FILENAME"] + ".h5"

    pipe = kp.Pipeline(timeit=args['--timeit'])
    pipe.attach(kp.io.OfflinePump, filename=args["FILENAME"])
    pipe.attach(km.StatusBar, every=1000)
    if args["--offline-header"]:
        pipe.attach(km.io.OfflineHeaderTabulator)
    if args["--event-info"]:
        pipe.attach(km.io.EventInfoTabulator)
    if args["--offline-hits"]:
        pipe.attach(km.io.HitsTabulator, kind="offline")
    if args["--online-hits"]:
        pipe.attach(km.io.HitsTabulator, kind="online")
    if args["--mc-hits"]:
        pipe.attach(km.io.HitsTabulator, kind="mc")
    if args["--mc-tracks"]:
        pipe.attach(km.io.MCTracksTabulator, read_usr_data=args["--mc-tracks-usr-data"])
    pipe.attach(blob_printer)
    pipe.attach(kp.io.HDF5Sink, filename=outfile)
    pipe.drain()
