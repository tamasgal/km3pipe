#!/usr/bin/env python3
# Filename: tres.py
"""
A tool to calculate and extract Cherenkov hit time residuals from reconstruced
events in offline ROOT files (currently only the JMuon chain).

Usage:
    tres [options] FILENAME
    tres (-h | --help)
    tres --version

Options:
    -o OUTFILE          Output file.
    --single-muon-only  Filter out single muon events
                        (only possible for RBR atmospheric muons - MUPAGE)
    -d DETX             Use a detector file to calibrate hits (not needed for
                        offline files created with km3net-dataformat
                        version 2+).
    -n N_EVENTS         Number of events to extract.
    -h --help           Show this screen.
    --version           Show the version.

"""


def run_pipeline(args):
    import km3pipe as kp
    import km3modules as km
    import km3io

    fname = args["FILENAME"]

    if args["-o"] is None:
        outfile = fname + ".tres.h5"
    else:
        outfile = args["-o"]

    def single_muon_filter(blob):
        """Only let events pass with exactly one muon"""
        if blob["event"].n_mc_tracks == 1:
            return blob

    def reco_filter(blob):
        """Filter out events without reco tracks"""
        if blob["event"].n_tracks > 0:
            return blob

    class TimeResidualsCalculator(kp.Module):
        def configure(self):
            detx = self.get("detx", default=None)
            self.calib = None

            if detx is not None:
                self.calib = kp.calib.Calibration(filename=detx)

        def process(self, blob):
            event = blob["event"]
            hits = event.hits

            if self.calib is not None:
                hits = self.calib.apply(hits)
                times = hits.time  # TODO: harmonise names
            else:
                times = hits.t  # TODO: harmonise names

            bt = km3io.tools.best_jmuon(event.tracks)
            cherenkov_params = kp.physics.cherenkov(hits, bt)

            t_res = kp.NDArray(
                times - cherenkov_params["t_photon"],
                h5loc="/t_res",
                title="Cherenkov hit time residuals",
            )
            blob["TRes"] = t_res

            return blob

    pipe = kp.Pipeline()
    pipe.attach(km.StatusBar, every=1000)
    pipe.attach(kp.io.OfflinePump, filename=fname)
    if args["--single-muon-only"]:
        pipe.attach(single_muon_filter)
    pipe.attach(reco_filter)
    pipe.attach(TimeResidualsCalculator, detx=args["-d"])
    pipe.attach(km.Keep, keys=["TRes"])
    pipe.attach(kp.io.HDF5Sink, filename=outfile)
    if args["-n"] is not None:
        pipe.drain(int(args["-n"]))
    else:
        pipe.drain()


def main():

    from docopt import docopt

    run_pipeline(docopt(__doc__))


if __name__ == "__main__":

    main()
