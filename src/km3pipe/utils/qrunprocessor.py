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
        -v PYTHONVENV  Path to the Python virtual env.
        -c CLUSTER     Cluster to run on (in2p3, woody, ...) [default: in2p3].
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

        def tqdm(x):
            return x

    cprint = kp.logger.get_printer("qrunprocessor")
    log = kp.logger.get_logger("qrunprocessor")

    RUN_LIST = os.path.abspath(args["RUN_LIST"])
    OUTPUT_PATH = os.path.abspath(args["OUTPUT_PATH"])
    SCRIPT = os.path.abspath(args["SCRIPT"])
    SUFFIX = args["-s"]
    DET_ID = int(args["DET_ID"])
    ET_PER_FILE = int(args["-e"]) * 60  # [s]
    FILES_PER_JOB = int(args["-n"])
    FSIZE = args["-f"]
    VMEM = args["-m"]
    LOG_PATH = os.path.abspath(args["-l"])
    JOB_NAME = args["-j"]
    DRYRUN = args["-q"]
    PYTHONVENV = os.path.abspath(args["-v"])
    CLUSTER = args["-c"]

    pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    with open(RUN_LIST, "r") as fobj:
        run_numbers = [int(run) for run in fobj.read().split()]

    xrootd_files = []
    for run in run_numbers:
        xrootd_path = kp.tools.xrootd_path(DET_ID, run)
        xrootd_files.append(xrootd_path)

    processed_files = [
        basename(f) for f in glob(join(OUTPUT_PATH, "*{}".format(SUFFIX)))
    ]

    rem_files = []
    for xrootd_file in xrootd_files:
        if basename(xrootd_file) + SUFFIX not in processed_files:
            rem_files.append(xrootd_file)

    cprint(
        "{} runs in total, {} already processed.".format(
            len(xrootd_files), len(processed_files)
        )
    )
    cprint(f"Proceeding with the remaining {len(rem_files)} files.")

    s = kp.shell.Script()

    for job_id, file_chunk in enumerate(chunks(rem_files, FILES_PER_JOB)):
        n_files = len(file_chunk)
        s.add(f"echo Creating run summary for {n_files} files")
        s.add("cd $TMPDIR; mkdir -p $USER; cd $USER")
        if PYTHONVENV is not None:
            s.add(". {}/bin/activate".format(PYTHONVENV))
        s.add("echo")

        for xpath in file_chunk:
            fname = basename(xpath)
            s.separator(" ")
            s.separator("=")
            s.echo(f"Processing {fname}:")
            s.add("pwd")
            s.add(f"xrdcp {xpath} {fname}")
            s.add(f"ls -al {fname}")
            s.add("km3pipe --version")
            s.add(f"KPrintTree -f {fname}")
            out_fname = fname + SUFFIX
            out_fpath = join(OUTPUT_PATH, out_fname)
            tmp_fname = out_fname + ".copying"
            tmp_fpath = join(OUTPUT_PATH, tmp_fname)
            s.add(f"{SCRIPT} {fname} -o {out_fname}")
            s.cp(out_fname, tmp_fpath)
            s.add(f"rm {out_fname}")
            s.mv(tmp_fpath, out_fpath)
            s.echo(f"File '{fname}' processed.")
            s.separator("-")

        walltime = time.strftime("%H:%M:%S", time.gmtime(ET_PER_FILE * n_files))

        kp.shell.qsub(
            s,
            "{}_{}".format(JOB_NAME, job_id),
            walltime=walltime,
            fsize=FSIZE,
            vmem=VMEM,
            log_path=LOG_PATH,
            xrootd=True,
            cluster=CLUSTER,
            dryrun=DRYRUN,
        )

        if DRYRUN:
            break

        s.clear()


if __name__ == "__main__":
    main()
