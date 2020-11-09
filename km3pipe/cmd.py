# Filename: cmd.py
"""
KM3Pipe command line utility.

Usage:
    km3pipe test
    km3pipe update [GIT_BRANCH]
    km3pipe createconf [--overwrite] [--dump]
    km3pipe retrieve DET_ID RUN [-i -o OUTFILE]
    km3pipe (-h | --help)
    km3pipe --version

Options:
    -h --help           Show this screen.
    -m                  Get the MC detector file (flips the sign of DET_ID).
    -c CALIBR_ID        Geometrical calibration ID (eg. A01466417)
    -i                  Use iRODS instead of xrootd to retrieve files.
    -o OUT              Output folder or filename.
    -t T0_SET           Time calibration ID (eg. A01466431)
    -s REGEX            Regular expression to filter the runsetup name/id.
    DET_ID              Detector ID (eg. D_ARCA001).
    DETECTOR            Detector (eg. ARCA).
    GIT_BRANCH          Git branch to pull (eg. develop).
    RUN                 Run number.

"""
import re
import sys
import subprocess
import os
from datetime import datetime
import time

import km3db

from . import version
from .tools import irods_path, xrootd_path
from .hardware import Detector

from km3pipe.logger import get_logger

log = get_logger(__name__)  # pylint: disable=C0103

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"

SPS_CACHE = "/sps/km3net/repo/data/cache"


def run_tests():
    import pytest
    import km3pipe

    pytest.main([os.path.dirname(km3pipe.__file__)])


def update_km3pipe(git_branch=""):
    print(
        "Running the km3pipe test suite. Make sure you have all the test "
        "dependencies and extras installed with\n\n"
        '    pip install "km3pipe[dev]"\n'
        '    pip install "km3pipe[extras]"\n'
    )
    if git_branch == "" or git_branch is None:
        git_branch = "master"
    os.system(
        "pip install -U git+http://git.km3net.de/km3py/km3pipe.git@{0}".format(
            git_branch
        )
    )


def retrieve(run_id, det_id, use_irods=False, out=None):
    """Retrieve run from iRODS for a given detector (O)ID"""
    det_id = int(det_id)

    if use_irods:
        rpath = irods_path(det_id, run_id)
        cmd = "iget -Pv"
    else:
        rpath = xrootd_path(det_id, run_id)
        cmd = "xrdcp"

    filename = os.path.basename(rpath)

    if out is not None and os.path.isdir(out):
        outfile = os.path.join(out, filename)
    elif out is None:
        outfile = filename
    else:
        outfile = out

    if os.path.exists(outfile):
        print("Output file '{}' already exists.".format(outfile))
        return

    cmd += " {0} {1}".format(rpath, outfile)

    if not km3db.core.on_whitelisted_host("lyon"):
        subprocess.call(cmd.split())
        return

    subfolder = os.path.join(*rpath.split("/")[-3:-1])
    cached_subfolder = os.path.join(SPS_CACHE, subfolder)
    cached_filepath = os.path.join(cached_subfolder, filename)
    lock_file = cached_filepath + ".in_progress"

    if os.path.exists(lock_file):
        print("File is already requested, waiting for the download to finish.")
        for _ in range(6 * 15):  # 15 minute timeout
            time.sleep(10)
            if not os.path.exists(lock_file):
                print("Done.")
                break
        else:
            print(
                "Timeout reached. Deleting the lock file and initiating a "
                "new download."
            )
            os.remove(lock_file)

    if not os.path.exists(cached_filepath):
        print("Downloading file to local SPS cache ({}).".format(SPS_CACHE))
        try:
            os.makedirs(cached_subfolder, exist_ok=True)
            subprocess.call(["touch", lock_file])
            subprocess.call(["chmod", "g+w", lock_file])
            subprocess.call(cmd.split())
            subprocess.call(["chmod", "g+w", outfile])
            subprocess.call(["cp", "-p", outfile, cached_filepath])
        except (OSError, KeyboardInterrupt) as e:
            print("\nAborting... {}".format(e))
            for f in [outfile, lock_file, cached_filepath]:
                subprocess.call(["rm", "-f", f])
            return
        finally:
            for f in [outfile, lock_file]:
                subprocess.call(["rm", "-f", f])

    subprocess.call(["ln", "-s", cached_filepath, outfile])


def main():
    from docopt import docopt

    args = docopt(
        __doc__,
        version="KM3Pipe {}".format(
            version,
        ),
    )

    if args["test"]:
        run_tests()

    if args["update"]:
        update_km3pipe(args["GIT_BRANCH"])

    if args["retrieve"]:
        retrieve(int(args["RUN"]), args["DET_ID"], use_irods=args["-i"], out=args["-o"])
