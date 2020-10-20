#!/usr/bin/env python
# coding=utf-8
# Filename: qrunqaqc.py
# Author: Tamas Gal <tgal@km3net.de>
"""
Submits jobs which analyse a batch of runs with ``JQAQC.sh`` to retrieve the
quality parameters. The obtained data is then optionally uploaded to the
runsummarynumbers table of the KM3NeT database.

The submitted run IDs are logged in ``BLACKLIST_{DET_ID}.txt`` and skipped
upon subsequential runs. The temporary quality parameters are stored in
the folder ``qparams`` and the job logs can be found in ``qlogs``.

To start from scratch, delete the above mentioned files. Runs which are were
already processed and present in the runsummarynumbers table are always
skipped.

Usage:
    qrunqaqc [options] DET_ID
    qrunqaqc (-h | --help)

Options:
    -b BATCH_SIZE  Number of runs to process in a single job.
    -j MAX_JOBS    Maximum number of jobs to submit [default: 100].
    -u             Upload summary data to the database automatically.
    -q             Dryrun, don't submit the jobs.
    -h --help      Show this screen.

"""
from collections import defaultdict
import math
import os
import subprocess
import time

import km3db
from tqdm import tqdm

import km3pipe as kp


ESTIMATED_TIME_PER_RUN = 60 * 10  # [s]


class QAQCAnalyser(object):
    """Determines  run quality parameters and uploads them to the DB"""

    def __init__(self, det_id, should_upload_to_db, log_file="qrunqaqc.log"):
        self.det_id = det_id
        self.should_upload_to_db = should_upload_to_db

        self.log = kp.logger.get_logger("qrunqaqc", filename=log_file)

        self.log.info("QAQC analysis started for detector ID %s", det_id)
        if should_upload_to_db:
            self.log.info(
                "The acquired data will be automatically updloaded to "
                "the database after each successful run processing."
            )

        self.jpp_version = kp.tools.get_jpp_version()
        if self.jpp_version is None:
            self.log.critical("Please load a Jpp environment")
            raise SystemExit()
        else:
            print("Run quality determination using Jpp '{}'".format(self.jpp_version))

        self.sds = km3db.StreamDS(container="pd")
        self.det_oid = km3db.tools.todetoid(det_id)

        self.runtable = self.sds.get("runs", detid=self.det_id)

        self.available_run_files = []

        cwd = os.getcwd()
        self.outdir = os.path.join(cwd, "qparams")
        self.jobdir = os.path.join(cwd, "qjobs")
        for path in [self.outdir, self.jobdir]:
            if not os.path.exists(path):
                os.makedirs(path)

        self.blacklist = os.path.join(cwd, "blacklist_{}.txt".format(self.det_id))

        self.stats = defaultdict(int)

        self._qparams = None
        self._columns = None
        self._blacklisted_run_ids = None

    def add_to_blacklist(self, run_id):
        """Put the run_id into a det_id specific blacklist"""
        with open(self.blacklist, "a") as fobj:
            fobj.write("{}\n".format(run_id))

    @property
    def blacklisted_run_ids(self):
        """The blacklisted run_ids"""
        if self._blacklisted_run_ids is None:
            if os.path.exists(self.blacklist):
                with open(self.blacklist) as fobj:
                    self._blacklisted_run_ids = set(int(l) for l in fobj.readlines())
            else:
                self._blacklisted_run_ids = set()
        return self._blacklisted_run_ids

    def retrieve_available_runs(self):
        files = kp.tools.ifiles("data/raw/sea/KM3NeT_{:08d}".format(self.det_id))
        self.available_run_files = {f.path: f.size for f in files}

    def run(self, batch_size, max_jobs, dryrun):
        """Walk through the runtable and submit batches of runs"""

        run_ids = self.runtable.RUN.values
        print("{} runs in total".format(len(run_ids)))

        try:
            already_processed_runs_ids = self.sds.get(
                "runsummarynumbers",
                detid=self.det_oid,
                minrun=min(run_ids),
                maxrun=max(run_ids),
                source_name=self.jpp_version,
                parameter_name="livetime_s",  # to lower the request size
            ).RUN.values
        except AttributeError:
            already_processed_runs_ids = set()
        else:
            print(
                "{} runs are already processed and available in the DB".format(
                    len(already_processed_runs_ids)
                )
            )

        if self.blacklisted_run_ids:
            print(
                "Skipping {} runs since they were already submitted "
                "and may be in the job queue. Delete the blacklist file "
                "to resubmit them next time this script is ran.".format(
                    len(self.blacklisted_run_ids)
                )
            )

        n_jobs = 0
        run_ids_to_process = []
        run_ids_to_check = sorted(
            set(run_ids) - set(already_processed_runs_ids) - self.blacklisted_run_ids,
            reverse=True,
        )

        print("Checking runs, retrieving file list from iRODS...")

        self.retrieve_available_runs()

        for run_id in tqdm(run_ids_to_check):
            if n_jobs >= max_jobs:
                self.log.warning(
                    "Maximum number of jobs reached (%s), "
                    "proceeding with batch submission",
                    max_jobs,
                )
                break
            self.log.info("Checking run '{}'".format(run_id))

            irods_path = kp.tools.irods_path(self.det_id, run_id)
            if irods_path in self.available_run_files.keys():
                run_ids_to_process.append(run_id)
                if batch_size and len(run_ids_to_process) % batch_size == 0:
                    n_jobs += 1
                continue
            else:
                self.log.info(
                    "  -> no file found on iRODS or an iRODS error "
                    "occured for run {}".format(run_id)
                )
                self.stats["Missing data or iRODS error"] += 1

        total_runs_to_process = len(run_ids_to_process)

        if total_runs_to_process == 0:
            print("No runs to process.")
        else:

            if batch_size is None:
                batch_size = int(math.ceil(total_runs_to_process / float(max_jobs)))

            print(
                "Proceeding with {} runs distributed over {} jobs, {} runs/job".format(
                    total_runs_to_process, max_jobs, batch_size
                )
            )

            run_id_chunks = kp.tools.chunks(run_ids_to_process, batch_size)

            self.pbar_runs = tqdm(
                total=len(run_ids_to_process), desc="Submitting runs", unit="run"
            )

            for run_ids in tqdm(run_id_chunks, desc="Jobs"):

                self.submit_batch(run_ids, dryrun=dryrun)

            self.pbar_runs.close()

        self.print_stats()

    def print_stats(self):
        """Print a summary of the collected statistics"""
        print("\n")
        print("Summary")
        print("=======")
        for key, value in self.stats.items():
            print("  {}: {}".format(key, value))

    def submit_batch(self, run_ids, dryrun):
        """Submit a QAQC.sh job for a given list of run IDs"""

        filesizes = []

        s = kp.shell.Script()
        for run_id in run_ids:
            self.log.info("adding run %s (det_id %s", run_id, self.det_id)
            xrootd_path = kp.tools.xrootd_path(self.det_id, run_id)
            self.log.info("xrootd path: %s", xrootd_path)
            size = self.available_run_files[kp.tools.irods_path(self.det_id, run_id)]
            self.log.info("filesize: %s (bytes)", size)
            filesizes.append(size)

            root_filename = os.path.basename(xrootd_path)

            out_filename = os.path.join(
                self.outdir, "{}_{}_qparams.csv".format(self.det_id, run_id)
            )
            t0set = self.runtable[self.runtable.RUN == run_id].T0_CALIBSETID.values[0]

            self.log.info("adding run '%s' with t0set '%s'", run_id, t0set)

            s.separator()
            s.echo("Processing run {}".format(run_id))
            s.separator("-")
            s.add("km3pipe detx {} -t {} -o d.detx".format(self.det_id, t0set))
            s.add("xrdcp {} {}".format(xrootd_path, root_filename))
            s.add("echo '{}'> {}".format(" ".join(self.columns), out_filename))
            s.add(
                "JQAQC.sh d.detx {} "
                "| awk '{{$1=$1;print}}' | tr '\\n' ' ' >> {}".format(
                    root_filename, out_filename
                )
            )
            if self.should_upload_to_db:
                s.add("streamds upload {}".format(out_filename))
            s.add("rm -f {}".format(root_filename))

            self.add_to_blacklist(run_id)
            self.pbar_runs.update(1)
            self.stats["Number of submitted runs"] += 1

        walltime = time.strftime(
            "%H:%M:%S", time.gmtime(ESTIMATED_TIME_PER_RUN * len(run_ids))
        )

        fsize = int(max(filesizes) / 1024 / 1024 * 1.1 + 100)

        identifier = "QAQC_{}_{}-{}".format(self.det_id, run_ids[0], run_ids[-1])

        job_script = kp.shell.qsub(
            s,
            identifier,
            vmem="4G",
            fsize="{}M".format(fsize),
            xrootd=True,
            walltime=walltime,
            silent=True,
            dryrun=dryrun,
        )
        self.log.info("  => job with %s runs submitted", len(run_ids))

        jobfile = os.path.join(self.jobdir, identifier + ".sh")
        with open(jobfile, "w") as fobj:
            fobj.write(job_script)
        self.log.info("     job file have been saved to {}".format(jobfile))
        if dryrun:
            self.stats["Number of dryrun jobs"] += 1
        else:
            self.stats["Number of submitted jobs"] += 1

    def retrieve_qparams(self):
        """Returns a list of quality parameters determined by JQAQC.sh"""
        command = "JQAQC.sh -h"
        try:
            qparams = subprocess.getoutput(command)
        except AttributeError:
            qparams = subprocess.check_output(command.split(), stderr=subprocess.STDOUT)
        return qparams.split("\n")[1].split()

    @property
    def columns(self):
        """Get the ordered columns for the quality parameter output"""
        if self._columns is None:
            qparams = self.retrieve_qparams()
            # Adapting to runsummarynumbers naming convention
            # Warning: for retrieving data, it's called "source_name"
            qparams[qparams.index("GIT")] = "source"
            qparams[qparams.index("detector")] = "det_id"

            self._columns = qparams
        return self._columns


def main():
    from docopt import docopt

    args = docopt(__doc__)

    try:
        batch_size = int(args["-b"])
    except TypeError:
        batch_size = None

    qaqc = QAQCAnalyser(det_id=int(args["DET_ID"]), should_upload_to_db=args["-u"])
    qaqc.run(batch_size=batch_size, max_jobs=int(args["-j"]), dryrun=bool(args["-q"]))


if __name__ == "__main__":
    main()
