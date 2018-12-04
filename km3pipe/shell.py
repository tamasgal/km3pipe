# Filename: shell.py
# pylint: disable=C0103
"""
Some shell helpers

"""
from __future__ import absolute_import, print_function, division

import os
import subprocess

from .tools import lstrip
from .logger import get_logger

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = get_logger(__name__)    # pylint: disable=C0103

JOB_TEMPLATE = lstrip(
    """
    #$ -N {job_name}
    #$ -M {email}
    ## Send mail at: start (b), completion (e), never (n)
    #$ -m {send_mail}
    #$ -j y
    #$ -o {log_path}/{job_name}{task_name}.log
    #$ -l os={platform}
    #$ -P P_{group}
    #$ -S {shell}
    {job_array_option}
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

    echo "========================================================"
    echo "Job started on" $(date)
    echo "========================================================"

    {script}

    echo "========================================================"
    echo "JAWOLLJA! Job exited on" $(date)
    echo "========================================================"
"""
)


def qsub(script, job_name, dryrun=False, *args, **kwargs):
    """Submit a job via qsub."""
    print("Preparing job script...")
    job_string = gen_job(script=script, job_name=job_name, *args, **kwargs)
    env = os.environ.copy()
    if dryrun:
        print(
            "This is a dry run! Here is the generated job file, which will "
            "not be submitted:"
        )
        print(job_string)
    else:
        print("Calling qsub with the generated job script.")
        p = subprocess.Popen(
            'qsub -V', stdin=subprocess.PIPE, env=env, shell=True
        )
        p.communicate(input=bytes(job_string.encode('ascii')))


def gen_job(
        script,
        job_name,
        log_path='qlogs',
        group='km3net',
        platform='cl7',
        walltime='00:10:00',
        vmem='8G',
        fsize='8G',
        shell=None,
        email=None,
        send_mail='n',
        job_array_start=1,
        job_array_stop=None,
        job_array_step=1,
        irods=False,
        sps=True,
        hpss=False,
        xrootd=False,
        dcache=False,
        oracle=False,
        split_array_logs=False
):
    """Generate a job script."""
    if shell is None:
        shell = os.environ['SHELL']
    if email is None:
        email = os.environ['USER'] + '@km3net.de'
    if isinstance(script, Script):
        script = str(script)
    log_path = os.path.join(os.getcwd(), log_path)
    if job_array_stop is not None:
        job_array_option = "#$ -t {}-{}:{}"  \
                           .format(job_array_start, job_array_stop,
                                   job_array_step)
    else:
        job_array_option = "#"
    if split_array_logs:
        task_name = '_$TASK_ID'
    else:
        task_name = ''
    job_string = JOB_TEMPLATE.format(
        script=script,
        email=email,
        send_mail=send_mail,
        log_path=log_path,
        job_name=job_name,
        group=group,
        walltime=walltime,
        vmem=vmem,
        fsize=fsize,
        irods=irods,
        sps=sps,
        hpss=hpss,
        xrootd=xrootd,
        dcache=dcache,
        oracle=oracle,
        shell=shell,
        platform=platform,
        job_array_option=job_array_option,
        task_name=task_name
    )
    return job_string


def hppsgrab(irod_path, method='irods'):
    """Generate command to download file from HPSS.

    Parameters
    ==========
    hpss_path: string
        Full path to resource, starting with ``"/in2p3.fr/group/km3net/"``
    method: string, optional [default: 'irods']
        Downloader client (``"irods"`` or ``"xrootd"``).
    """
    METHODS = {
        'irods': {
            'getter': 'iget -K -P -N0 -f ',
            'prefix': '',
        },
        'xrootd': {
            'getter': 'xrdcp -f ',
            'prefix': 'root://ccxroot.in2p3.fr:1999//hpss',
        }
    }
    meth = METHODS[method]
    getter = meth['getter']
    prefix = meth['prefix']
    cmd = getter + prefix
    return cmd


def get_jpp_env(jpp_dir):
    """Return the environment dict of a loaded Jpp env.

    The returned env can be passed to `subprocess.Popen("J...", env=env)`
    to execute Jpp commands.

    """
    env = {
        v[0]: ''.join(v[1:])
        for v in [
            l.split('=') for l in os.popen(
                "source {0}/setenv.sh {0} && env".format(jpp_dir)
            ).read().split('\n') if '=' in l
        ]
    }
    return env


class Script(object):
    """A shell script which can be built line by line for `qsub`."""

    def __init__(self):
        self.lines = []

    def add(self, line):
        """Add a new line"""
        self.lines.append(line)

    def echo(self, text):
        """Add an echo command. The given text will be double qouted."""
        self.lines.append('echo "{}"'.format(text))

    def separator(self, character='=', length=42):
        """Add a visual separator."""
        self.echo(character * length)

    def cp(self, source, target):
        """Add a new copy instruction"""
        self._add_two_argument_command('cp', source, target)

    def mv(self, source, target):
        """Add a new move instruction"""
        self._add_two_argument_command('mv', source, target)

    def mkdir(self, folder_path):
        """Add a new 'mkdir -p' instruction"""
        self.add('mkdir -p "{}"'.format(folder_path))

    def iget(self, irods_path):
        """Add an iget command to retrieve a file from iRODS."""
        self.add('iget -v "{}"'.format(irods_path))

    def _add_two_argument_command(self, command, arg1, arg2):
        """Helper function for two-argument commands"""
        self.lines.append("{} {} {}".format(command, arg1, arg2))

    def clear(self):
        self.lines = []

    def __add__(self, other):
        new_script = Script()
        new_script.lines = self.lines + other.lines
        return new_script

    def __str__(self):
        return '\n'.join(self.lines)

    def __repr__(self):
        return "# Shell script\n" + str(self)
