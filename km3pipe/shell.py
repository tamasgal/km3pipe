# coding=utf-8
# cython: profile=True
# Filename: shell.py
# cython: embedsignature=True
# pylint: disable=C0103
"""
Some shell helpers

"""
from __future__ import division, absolute_import, print_function

import os
import subprocess

from .tools import lstrip
from .logger import logging

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = logging.getLogger(__name__)  # pylint: disable=C0103


JOB_TEMPLATE = lstrip("""
    ########################################################
    #$ -N {job_name}
    #$ -M {email}
    ## Send mail at: start (b), completion (e), never (n)
    #$ -m {send_mail}
    #$ -j y
    #$ -o {log_path}/{job_name}.log
    #$ -l os={platform}
    #$ -P P_{group}
    #$ -S {shell}
    #
    ## Walltime (HH:MM:SS)
    #$ -l ct={walltime}
    ## Memory (Units: G, M, K, B; Min 64M)
    #$ -l vmem={vmem}
    ## Local scratch diskspace (at TMPDIR)
    #$ -l fsize={fsize}
    #
    ## Extra Resources
    #$ -l irods={irods:d}
    #$ -l sps={sps:d}
    #$ -l hpss={hpss:d}
    #$ -l xrootd={xrootd:d}
    #$ -l dcache={dcache:d}
    #$ -l oracle={oracle:d}
    ########################################################

    echo "========================================================"
    echo "Job started on" $(date)
    echo "========================================================"

    {script}

    echo "========================================================"
    echo "Job exited on" $(date)
    echo "========================================================"
""")


def qsub(script, job_name, log_path='qlogs', group='km3net', platform='cl7',
         walltime='00:10:00', vmem='8G', fsize='8G', shell=os.environ['SHELL'],
         email=os.environ['USER']+'@km3net.de', send_mail='n',
         irods=False, sps=True, hpss=False, xrootd=False,
         dcache=False, oracle=False,
         dryrun=False):
    """Submit a job via qsub."""
    print("Preparing job script...")
    if isinstance(script, Script):
        script = str(script)
    log_path = os.path.join(os.getcwd(), log_path)
    job_string = JOB_TEMPLATE.format(
            script=script, email=email, send_mail=send_mail, log_path=log_path,
            job_name=job_name, group=group, walltime=walltime, vmem=vmem,
            fsize=fsize, irods=irods, sps=sps, hpss=hpss, xrootd=xrootd,
            dcache=dcache, oracle=oracle, shell=shell, platform=platform)
    env = os.environ.copy()
    if dryrun:
        print("This is a dry run! Here is the generated job file, which will "
              "not be submitted:")
        print(job_string)
    else:
        print("Calling qsub with the generated job script.")
        p = subprocess.Popen('qsub -V', stdin=subprocess.PIPE, env=env,
                             shell=True)
        p.communicate(input=bytes(job_string.encode('ascii')))


def get_jpp_env(jpp_dir):
    """Return the environment dict of a loaded Jpp env.

    The returned env can be passed to `subprocess.Popen("J...", env=env)`
    to execute Jpp commands.

    """
    env = {v[0]: ''.join(v[1:]) for v in
           [l.split('=') for l in
            os.popen("source {0}/setenv.sh {0} && env"
                     .format(jpp_dir)).read().split('\n')
            if '=' in l]}
    return env


class Script(object):
    """A shell script which can be built line by line for `qsub`."""
    def __init__(self):
        self.lines = []

    def add(self, line):
        """Add a new line"""
        self.lines.append(line)

    def clear(self):
        self.lines = []

    def __str__(self):
        return '\n'.join(self.lines)

    def __repr__(self):
        return "# Shell script\n" + str(self)
