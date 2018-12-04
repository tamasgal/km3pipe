#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: ts=4 sw=4 et
"""
================================
K40 Calibration Batch Processing
================================

Standalone job submitter for K40 offline calibrations with KM3Pipe.

.. code-block:: console

    Usage:
        qk40calib.py OUTPUT_PATH [options]
        qk40calib.py (-h | --help)

    Options:
        OUTPUT_PATH  Folder to store the calibration data.
        -d DET_ID    Detector ID [default: 29].
        -t TMAX      Coincidence time window [default: 10].
        -n N_RUNS    Number of runs to process per job [default: 10].
        -e ET        Estimated walltime per run in minutes [default: 8].
        -m VMEM      Estimated vmem for a job [default: 8G].
        -s RUNSETUP  Match [default: PHYS.1710v5-TUNED.HRV19.3D_T_S_MX.NBMODULE].
        -j JOBNAME   The name of the submitted jobs [default: k40calib].
        -l LOG_PATH  Path of the job log files [default: qlogs].
        -q           Dryrun: don't submit jobs, just print the first job script.
        -h --help    Show this screen.

"""
from __future__ import absolute_import, print_function, division

import os
import re
from glob import glob
import time
from km3pipe.shell import qsub, Script
import km3pipe as kp
from docopt import docopt


def main():
    args = docopt(__doc__)

    DET_ID = int(args['-d'])
    TMAX = int(args['-t'])
    ET_PER_RUN = int(args['-e']) * 60    # [s]
    RUNS_PER_JOB = int(args['-n'])
    VMEM = args['-m']
    CWD = os.getcwd()
    LOG_PATH = args['-l']
    JOB_NAME = args['-j']
    CALIB_PATH = os.path.join(CWD, args['OUTPUT_PATH'])
    RUN_SUBSTR = args['-s']
    DRYRUN = args['-q']

    if not os.path.exists(CALIB_PATH):
        os.makedirs(CALIB_PATH)

    db = kp.db.DBManager()
    run_table = db.run_table(det_id=DET_ID)
    phys_run_table = run_table[run_table.RUNSETUPNAME.str.contains(RUN_SUBSTR)]
    phys_runs = set(phys_run_table.RUN)
    processed_runs = set(
        int(re.search("_\\d{8}_(\\d{8})", s).group(1))
        for s in glob(os.path.join(CALIB_PATH, '*.k40_cal.p'))
    )
    remaining_runs = list(phys_runs - processed_runs)
    print("Remaining runs: {}".format(remaining_runs))
    s = Script()

    for job_id, runs_chunk in enumerate(kp.tools.chunks(remaining_runs,
                                                        RUNS_PER_JOB)):
        n_runs = len(runs_chunk)
        print(
            "Preparing batch script for a chunk of {} runs.".format(
                len(runs_chunk)
            )
        )
        s.add("cd $TMPDIR; mkdir -p $USER; cd $USER")
        for run in runs_chunk:
            s.add("echo Processing {}:".format(run))
            irods_path = kp.tools.irods_filepath(DET_ID, run)
            root_filename = os.path.basename(irods_path)
            calib_filename = root_filename + '.k40_cal.p'
            s.add("iget -v {}".format(irods_path))
            s.add(
                "CTMIN=$(JPrint -f {}|grep '^ctMin'|awk '{{print $2}}')".
                format(root_filename)
            )
            s.add(
                "k40calib {} {} -t {} -c $CTMIN -o {}".format(
                    root_filename, DET_ID, TMAX, calib_filename
                )
            )
            s.add("cp {} {}".format(calib_filename, CALIB_PATH))
            s.add("rm -f {}".format(root_filename))
            s.add("rm -f {}".format(calib_filename))
            s.add("echo Run {} processed.".format(run))
            s.add("echo " + 42 * "=")

        walltime = time.strftime('%H:%M:%S', time.gmtime(ET_PER_RUN * n_runs))
        qsub(
            s,
            '{}_{}'.format(JOB_NAME, job_id),
            walltime=walltime,
            vmem=VMEM,
            log_path=LOG_PATH,
            irods=True,
            dryrun=DRYRUN
        )

        if DRYRUN:
            break

        s.clear()


if __name__ == '__main__':
    main()
