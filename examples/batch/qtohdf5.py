#!/usr/bin/env python
# Filename: qtohdf5.py
# Author: Tamas Gal <tgal@km3net.de>
"""
======================================
Convert ROOT to HDF5 on the Batch Farm
======================================

Convert ROOT files to HDF5 on the SGE.

The ``IRODS_PATH`` will be scanned recursively and if ``-s`` is specified, a
substring will be used to filter the files. You can for example convert
all numu CC files in ``/in2p3/km3net/mc/prod/v4/JTE_r2356`` if you call
the script with ``-s numuCC``.

Before constructing the job scripts, the ``OUTPUT_PATH`` will be traversed
to find files which have already been converted to avoid multiple conversions.

.. code-block:: console

    Usage:
        qtohdf5.py IRODS_PATH OUTPUT_PATH [options]
        qtohdf5.py (-h | --help)

    Options:
        IRODS_PATH     Path the the files on iRODS.
        OUTPUT_PATH    Path to store the converted HDF5 files.
        -s SUBSTRING   String to match on filenames.
        -n N_FILES     Number of files to process per job [default: 10].
        -e ET          Estimated walltime per file in minutes [default: 15].
        -m VMEM        Estimated vmem for a job [default: 8G].
        -j JOBNAME     The name of the submitted jobs [default: tohdf5].
        -l LOG_PATH    Path of the job log files [default: qlogs].
        -q             Dryrun: don't submit jobs, just print the first job script.
        -h --help      Show this screen.

"""
from __future__ import absolute_import, print_function, division

from glob import glob
import os
from os.path import basename, join
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
    IRODS_PATH = args['IRODS_PATH']
    OUTPUT_PATH = join(CWD, args['OUTPUT_PATH'])
    SUBSTRING = '' if args['-s'] is None else args['-s']
    ET_PER_FILE = int(args['-e']) * 60    # [s]
    FILES_PER_JOB = int(args['-n'])
    VMEM = args['-m']
    LOG_PATH = args['-l']
    JOB_NAME = args['-j']
    DRYRUN = args['-q']

    mkdir(OUTPUT_PATH)

    mupage_files = kp.tools.ifiles(args['IRODS_PATH'])
    files = [basename(f) for f in mupage_files if SUBSTRING in f]
    conv_files = [basename(f)[:-3] for f in glob(join(OUTPUT_PATH, '*.h5'))]
    rem_files = list(set(files) - set(conv_files))

    print("{} MUPAGE files on iRODS".format(len(files)))
    print("{} files to be converted.".format(len(rem_files)))

    s = kp.shell.Script()

    for job_id, file_chunk in enumerate(chunks(rem_files, FILES_PER_JOB)):
        n_files = len(file_chunk)
        s.add("echo A job to convert {} files to HDF5".format(n_files))
        s.add("cd $TMPDIR; mkdir -p $USER; cd $USER")
        s.add("echo")

        for fname in file_chunk:
            ipath = join(IRODS_PATH, fname)
            s.separator('=')
            s.echo("Processing {}:".format(fname))
            s.iget(ipath)
            h5_fname = fname + '.h5'
            lock_fname = join(OUTPUT_PATH, h5_fname + '.copying')
            s.add("tohdf5 {} -o {}".format(fname, h5_fname))
            s.add("touch {}".format(lock_fname))
            s.cp(h5_fname, OUTPUT_PATH)
            s.add("rm -f {}".format(lock_fname))
            s.add("rm -f {}".format(h5_fname))
            s.echo("File '{}' converted.".format(fname))
            s.separator('-')

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
