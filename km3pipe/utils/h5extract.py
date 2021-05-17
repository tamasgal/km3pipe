#!/usr/bin/env python3
# Filename: h5extract.py
"""
A tool to extract data from KM3NeT ROOT files to HDF5.

Usage:
    h5extract [options] FILENAME
    h5extract (-h | --help)
    h5extract --version

Options:
    -o OUTFILE                  Output file.
    -n N_EVENTS                 Number of events to extract.
    --offline-header            The header of an offline file.
    --event-info                General event information.
    --offline-hits              Offline hits.
    --with-calibration          Include calibration information for offline hits.
    --mc-hits                   MC hits (use with care!).
    --online-hits               Snapshot and triggered hits (combined).
    --mc-tracks                 MC tracks..
    --mc-tracks-usr-data        "usr" data from MC tracks (this will be slow).
    --reco-tracks               Reconstructed tracks.
    --best-tracks               Write a separate best track dataset for each known reco.
    --provenance-file=FILENAME  The file to store the provenance information.
    --timeit                    Print detailed pipeline performance statistics.
    --step-size=N               Number of events to cache or amount of data [default: 2000].
    --aashower-legacy           Use only with old MC files where aashower rec_stage=100 (now it is 300)
    -h --help                   Show this screen.
    --version                   Show the version.

"""
from thepipe import Provenance
import km3pipe as kp
import km3modules as km


def main():
    from docopt import docopt

    args = docopt(__doc__, version=kp.version)
    step_size = int(args["--step-size"])

    default_flags = (
        "--offline-header",
        "--event-info",
        "--offline-hits",
        "--mc-hits",
        "--mc-tracks",
        "--with-calibration",
        "--reco-tracks",
        "--best-tracks",
    )
    if not any([args[k] for k in default_flags]):
        for k in default_flags:
            args[k] = True

    outfile = args["-o"]
    if outfile is None:
        outfile = args["FILENAME"] + ".h5"

    provfile = args["--provenance-file"]
    if provfile is None:
        provfile = outfile + ".prov.json"

    Provenance().outfile = provfile
    pipe = kp.Pipeline(timeit=args["--timeit"])
    pipe.attach(kp.io.OfflinePump, filename=args["FILENAME"], step_size=step_size)
    pipe.attach(km.StatusBar, every=1000)
    pipe.attach(km.common.MemoryObserver, every=500)
    if args["--offline-header"]:
        pipe.attach(km.io.OfflineHeaderTabulator)
    if args["--event-info"]:
        pipe.attach(km.io.EventInfoTabulator)
    if args["--offline-hits"]:
        pipe.attach(
            km.io.HitsTabulator,
            name="Offline",
            kind="offline",
            with_calibration=args["--with-calibration"],
        )
    if args["--online-hits"]:
        pipe.attach(km.io.HitsTabulator, name="Online", kind="online")
    if args["--mc-hits"]:
        pipe.attach(km.io.HitsTabulator, name="MC", kind="mc")
    if args["--mc-tracks"]:
        pipe.attach(km.io.MCTracksTabulator, read_usr_data=args["--mc-tracks-usr-data"])
    if args["--reco-tracks"]:
        pipe.attach(
            km.io.RecoTracksTabulator,
            best_tracks=args["--best-tracks"],
            aashower_legacy=args["--aashower-legacy"],
        )
    pipe.attach(kp.io.HDF5Sink, filename=outfile)
    if args["-n"] is not None:
        pipe.drain(int(args["-n"]))
    else:
        pipe.drain()
