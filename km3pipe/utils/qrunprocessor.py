#!/usr/bin/env python
# Filename: qrunprocessor.py
# Author: Tamas Gal <tgal@km3net.de>
"""
=========================================
Create a summary for a given list of runs
=========================================

Use this batch runner to process a given list of run numbers with a
script, which takes a `-o` to create a summary file, which has the name
of the processed file + a given suffix.
The actual call will look like this, with a bit of copy safeness:

    /abs/path/to/SCRIPT run.root -o /abs/path/to/OUTPUT_PATH+SUFFIX

Before constructing the job scripts, the ``OUTPUT_PATH`` will be traversed
to find files which have already been converted to avoid multiple conversions.

.. code-block:: console

    Usage:
        qrunprocessor [options] DET_ID RUN_LIST OUTPUT_PATH SCRIPT
        qrunprocessor (-h | --help)

    Options:
        DET_ID         Detector ID (e.g. 29).
        RUN_LIST       Path to the file containing the space separated run IDs.
        OUTPUT_PATH    Path to store the individual summary files.
        SCRIPT         The script to fire up.
        -s SUFFIX      The suffix, appended by SCRIPT [default: .summary.h5].
        -n N_FILES     Number of files to process per job [default: 10].
        -e ET          Estimated walltime per file in minutes [default: 15].
        -f FSIZE       Estimated filesystem size for a job [default: 12G].
        -m VMEM        Estimated vmem for a job [default: 8G].
        -j JOBNAME     The name of the submitted jobs [default: qrunprocessor].
        -l LOG_PATH    Path of the job log files [default: qlogs].
        -q             Dryrun: don't submit jobs, just print the job script.
        -h --help      Show this screen.

"""
__author__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__version__ = "1.0"


def main():
    from docopt import docopt
    args = docopt(__doc__, version=__version__)

    from glob import glob
    import os
    from os.path import basename, join, abspath
    import pathlib
    import time
    import km3pipe as kp
    from km3pipe.tools import chunks, iexists

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x: x

    cprint = kp.logger.get_printer('qrunprocessor')
    log = kp.logger.get_logger('qrunprocessor')

    CWD = os.getcwd()
    RUN_LIST = join(CWD, args['RUN_LIST'])
    OUTPUT_PATH = join(CWD, args['OUTPUT_PATH'])
    SCRIPT = abspath(args['SCRIPT'])
    SUFFIX = args['-s']
    DET_ID = int(args['DET_ID'])
    ET_PER_FILE = int(args['-e']) * 60    # [s]
    FILES_PER_JOB = int(args['-n'])
    FSIZE = args['-f']
    VMEM = args['-m']
    LOG_PATH = args['-l']
    JOB_NAME = args['-j']
    DRYRUN = args['-q']

    pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    with open(RUN_LIST, 'r') as fobj:
        run_numbers = [int(run) for run in fobj.read().split()]

    irods_files = []
    for run in run_numbers:
        irods_path = kp.tools.irods_filepath(DET_ID, run)
        irods_files.append(irods_path)

    processed_files = [
        basename(f) for f in glob(join(OUTPUT_PATH, '*{}'.format(SUFFIX)))
    ]

    rem_files = []
    for irods_file in irods_files:
        if basename(irods_file) + SUFFIX not in processed_files:
            rem_files.append(irods_file)

    cprint("Checking if files are accessible on iRODS")
    missing_files_on_irods = 0
    for rem_file in tqdm(rem_files):
        if not iexists(rem_file):
            log.error(
                "Skipping file, since not found on iRODS: {}".format(rem_file)
            )
            rem_files.remove(rem_file)
            missing_files_on_irods += 1

    cprint(
        "{} runs in total, {} already processed ({} missing on iRODS).".format(
            len(irods_files), len(processed_files), missing_files_on_irods
        )
    )
    cprint("Proceeding with the remaining {} files.".format(len(rem_files)))

    s = kp.shell.Script()

    for job_id, file_chunk in enumerate(chunks(rem_files, FILES_PER_JOB)):
        n_files = len(file_chunk)
        s.add("echo Creating run summary for {} files".format(n_files))
        s.add("cd $TMPDIR; mkdir -p $USER; cd $USER")
        s.add("echo")

        for ipath in file_chunk:
            fname = basename(ipath)
            s.separator(' ')
            s.separator('=')
            s.echo("Processing {}:".format(fname))
            s.add('pwd')
            s.iget(ipath)
            s.add('ls -al {}'.format(fname))
            s.add('JPrintTree -f {}'.format(fname))
            out_fname = fname + SUFFIX
            out_fpath = join(OUTPUT_PATH, out_fname)
            tmp_fname = out_fname + '.copying'
            tmp_fpath = join(OUTPUT_PATH, tmp_fname)
            s.add("{} {} -o {}".format(SCRIPT, fname, out_fname))
            s.cp(out_fname, tmp_fpath)
            s.mv(tmp_fpath, out_fpath)
            s.echo("File '{}' processed.".format(fname))
            s.separator('-')

        walltime = time.strftime(
            '%H:%M:%S', time.gmtime(ET_PER_FILE * n_files)
        )

        kp.shell.qsub(
            s,
            '{}_{}'.format(JOB_NAME, job_id),
            walltime=walltime,
            fsize=FSIZE,
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
