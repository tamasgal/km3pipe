#!/bin/bash
set -e

for file in $(python -m km3net_testdata offline)/*.root; do
    echo "Processing $file"
    echo "  two events"
    h5extract -n 2 $file

    # -o OUTFILE                  Output file.
    # -n N_EVENTS                 Number of events to extract.
    # --offline-header            The header of an offline file.
    # --event-info                General event information.
    # --offline-hits              Offline hits.
    # --mc-hits                   MC hits (use with care!).
    # --online-hits               Snapshot and triggered hits (combined).
    # --mc-tracks                 MC tracks..
    # --mc-tracks-usr-data        "usr" data from MC tracks (this will be slow).
    # --reco-tracks               Reconstructed tracks.
    # --provenance-file=FILENAME  The file to store the provenance information.
    # --timeit                    Print detailed pipeline performance statistics.
    # --step-size=N               Number of events to cache or amount of data [default: 2000].

    echo done
done

