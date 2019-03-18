#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: qprefit.py
# Author: Tamas Gal <tgal@km3net.de>
"""
======================
ROyPrefit batch script
======================

Apply the ROyPrefit to a set of files.

Usage:
    qprefit.py PATH OUTPUT_PATH [options] FITTER
    qprefit.py (-h | --help)

Options:
    PATH         Path to the files.
    FITTER       The fitter script, which will be called with the filepath.
    OUTPUT_PATH  Path to store the prefit summary CSV files.
    -x SUFFIX    Suffix to append to the output file [default: .royprefit.csv].
    -n N_FILES   Number of files to process per job [default: 20].
    -e ET        Estimated walltime per file in minutes [default: 3].
    -m VMEM      Estimated vmem for a job [default: 8G].
    -j JOBNAME   The name of the submitted jobs [default: royprefit].
    -l LOG_PATH  Path of the job log files [default: qlogs].
    -q           Dryrun: don't submit jobs, just print the first job script.
    -h --help    Show this screen.

"""
from __future__ import absolute_import, print_function, division

from glob import glob
import os
from os.path import basename, join, abspath
import time
import km3pipe as kp
from km3pipe.tools import chunks

__author__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__version__ = "1.0"


def mkdir(path):
    """Create folder if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    from docopt import docopt
    args = docopt(__doc__, version=__version__)

    CWD = os.getcwd()
    FITTER = abspath(args['FITTER'])
    PATH = abspath(args['PATH'])
    OUTPUT_PATH = join(CWD, args['OUTPUT_PATH'])
    SUFFIX = args['-x']
    ET_PER_FILE = int(args['-e']) * 60    # [s]
    FILES_PER_JOB = int(args['-n'])
    VMEM = args['-m']
    LOG_PATH = args['-l']
    JOB_NAME = args['-j']
    DRYRUN = args['-q']

    mkdir(OUTPUT_PATH)

    files = glob(join(PATH, "*.h5"))
    summaries = [
        basename(f)[:-len(SUFFIX)]
        for f in glob(join(OUTPUT_PATH, '*' + SUFFIX))
    ]
    rem_files = list(set(basename(f) for f in files) - set(summaries))

    print("{} files in total".format(len(files)))
    print("{} files to be fitted.".format(len(rem_files)))

    s = kp.shell.Script()

    for job_id, file_chunk in enumerate(chunks(rem_files, FILES_PER_JOB)):
        n_files = len(file_chunk)
        s.add("echo 'ROyPrefitting {} files'".format(n_files))
        s.add("cd $TMPDIR; mkdir -p $USER; cd $USER")
        s.add("echo")

        for fname in file_chunk:
            fpath = join(PATH, fname)
            oname = fname + SUFFIX
            opath = join(OUTPUT_PATH, oname)
            s.add("echo '" + 42 * "=" + "'")
            s.add("echo Processing {}:".format(fname))
            s.add("{} {}".format(FITTER, fpath))
            s.add("cp {} {}".format(oname, opath))
            s.add("echo File '{}' fitted.".format(fname))
            s.add("echo '" + 42 * "=" + "'")

        walltime = time.strftime(
            '%H:%M:%S', time.gmtime(ET_PER_FILE * n_files)
        )

        kp.shell.qsub(
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


if __name__ == "__main__":
    main()
