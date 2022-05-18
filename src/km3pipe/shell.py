# Filename: shell.py
# pylint: disable=C0103
"""
Some shell helpers

"""
import os
import subprocess
from warnings import warn

from .tools import lstrip
from .logger import get_logger

from subprocess import DEVNULL  # py3k

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = get_logger(__name__)  # pylint: disable=C0103

BATCH_TYPE = {"in2p3": "slurm", "woody": "torque"}
SUBMIT_CMD = {"slurm": "squeue", "torque": "qsub"}

JOB_TEMPLATES = {
    "in2p3": lstrip(
        """
        #SBATCH --partition=htc
        #SBATCH --job-name={job_name}
        #SBATCH --mail-user= {email}
        #SBATCH --mail-type {send_mail}
        #SBATCH --output={log_path}/{job_name}{task_name}.log
        #SBATCH --account={group}
        #
        ## Walltime (HH:MM:SS), for different batch systems (IN2P3, ECAP, ...)
        #SBATCH --time={walltime}
        ## Memory (Units: G, M, K, B; Min 64M)
        #SBATCH --mem={memory}
        #
        ## Extra Resources (sps, irods, hpss, oracle, xrootd, dcache)
        #SBATCH --licenses={resources}
        {extra_options}

        echo "========================================================"
        echo "Job started on" $(date)
        echo "========================================================"

        {script}

        echo "========================================================"
        echo "JAWOLLJA! Job exited on" $(date)
        echo "========================================================"
        """
    ),
    "woody": lstrip(
        """
        #PBS -N {job_name}
        #PBS -M {email} -m a
        #PBS -o {log_path}/{job_name}{task_name}.out.log
        #PBS -e {log_path}/{job_name}{task_name}.err.log
        #PBS -l nodes={nodes}:ppn={ppn}{node_type}
        #PBS -l walltime={walltime}

        echo "========================================================"
        echo "Job started on" $(date)
        echo "========================================================"

        {script}

        echo "========================================================"
        echo "JAWOLLJA! Job exited on" $(date)
        echo "========================================================"
        """
    ),
}


def qsub(script, job_name, dryrun=False, silent=False, *args, **kwargs):
    """
    Submit a job via qsub.

    Returns the job script as string.
    """
    warn(
        "qsub is deprecated and will be removed in the next major version!",
        DeprecationWarning,
        stacklevel=2,
    )
    submit(script, job_name, dryrun=dryrun, silent=silent, *arg, **kwargs)


def submit(script, job_name, dryrun=False, silent=False, *args, **kwargs):
    """
    Submit a job.

    Returns the job script as string.
    """
    submit_cmd = SUBMIT_CMD[BATCH_TYPE[kwargs["cluster"]]]

    if not silent:
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
        if not silent:
            print("Calling {} with the generated job script.".format(submit_cmd))
            out_pipe = subprocess.PIPE
        else:
            out_pipe = DEVNULL
        p = subprocess.Popen(
            "{} -V".format(submit_cmd),
            stdin=subprocess.PIPE,
            env=env,
            shell=True,
            stdout=out_pipe,
        )
        p.communicate(input=bytes(job_string.encode("ascii")))

    return job_string


def _gen_job_slurm(**kwargs):
    resources = []
    if kwargs["sps"]:
        resources.append("sps")
    if kwargs["irods"]:
        resources.append("irods")
    if kwargs["hpss"]:
        resources.append("hpss")
    if kwargs["xrootd"]:
        resources.append("xrootd")
    if kwargs["dcache"]:
        resources.append("dcache")
    if kwargs["oracle"]:
        resources.append("oracle")
    resources = ",".join(resources)

    email = (
        os.environ["USER"] + "@km3net.de"
        if kwargs["email"] is None
        else kwargs["email"]
    )
    del kwargs["kwargs"]

    log_path = os.path.abspath(kwargs["log_path"])
    del kwargs["log_path"]

    if job_array_stop is not None:
        job_array_option = "#SBATCH --array {}-{}:{}".format(
            kwargs["job_array_start"],
            kwargs["job_array_stop"],
            kwargs["job_array_step"],
        )
    else:
        job_array_option = "#"

    extra_options = "\n".join([job_array_option])

    job_string = JOB_TEMPLATES[kwargs["cluster"]].format(
        resources=resources,
        email=email,
        extra_options=extra_options,
        log_path=log_path,
        **kwargs
    )

    return job_string


def _gen_job_torque(**kwargs):
    """Generate a job script."""
    shell = os.environ["SHELL"] if kwargs["shell"] is None else kwargs["shell"]
    del kwargs["shell"]

    email = (
        os.environ["USER"] + "@km3net.de"
        if kwargs["email"] is None
        else kwargs["email"]
    )
    del kwargs["kwargs"]

    cpu = kwargs["walltime"] if kwargs["walltime"] is None else kwargs["cpu"]
    del kwargs["walltime"]

    script = (
        str(kwargs["script"])
        if isinstance(kwargs["script"], Script)
        else kwargs["script"]
    )
    del kwargs["script"]

    log_path = os.path.abspath(kwargs["log_path"])
    del kwargs["log_path"]

    if job_array_stop is not None:
        job_array_option = "#$ -t {}-{}:{}".format(
            kwargs["job_array_start"],
            kwargs["job_array_stop"],
            kwargs["job_array_step"],
        )
    else:
        job_array_option = "#"

    if kwargs["split_array_logs"]:
        task_name = "_$TASK_ID"
    else:
        task_name = ""

    if kwargs["node_type"] is not None:
        node_type = ":" + str(kwargs["node_type"])
    else:
        node_type = ""
    del kwargs["node_type"]

    job_string = JOB_TEMPLATES[kwargs["cluster"]].format(
        email=email,
        log_path=log_path,
        cpu=cpu,
        job_array_option=job_array_option,
        task_name=task_name,
        node_type=node_type,
        **kwargs
    )
    return job_string


def gen_job(
    script,
    job_name,
    log_path="qlogs",
    group="km3net",
    walltime="00:10:00",
    nodes=1,
    ppn=4,
    node_type=None,
    cluster="in2p3",
    memory="3G",
    email=None,
    send_mail="n",
    job_array_start=1,
    job_array_stop=None,
    job_array_step=1,
    irods=False,
    sps=True,
    hpss=False,
    xrootd=False,
    dcache=False,
    oracle=False,
    split_array_logs=False,
):
    kwargs = locals().items()
    if BATCH_TYPE[cluster] == "slurm":
        return _gen_job_slurm(**kwargs)
    elif BATCH_TYPE[cluster] == "torque":
        return _gen_job_torque(**kwargs)


def get_jpp_env(jpp_dir):
    """Return the environment dict of a loaded Jpp env.

    The returned env can be passed to `subprocess.Popen("J...", env=env)`
    to execute Jpp commands.

    """
    env = {
        v[0]: "".join(v[1:])
        for v in [
            l.split("=")
            for l in os.popen("source {0}/setenv.sh {0} && env".format(jpp_dir))
            .read()
            .split("\n")
            if "=" in l
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

    def separator(self, character="=", length=42):
        """Add a visual separator."""
        self.echo(character * length)

    def cp(self, source, target):
        """Add a new copy instruction"""
        self._add_two_argument_command("cp", source, target)

    def mv(self, source, target):
        """Add a new move instruction"""
        self._add_two_argument_command("mv", source, target)

    def mkdir(self, folder_path):
        """Add a new 'mkdir -p' instruction"""
        self.add('mkdir -p "{}"'.format(folder_path))

    def iget(self, irods_path, attempts=1, pause=15):
        """Add an iget command to retrieve a file from iRODS.

        Parameters
        ----------
            irods_path: str
                Filepath which should be fetched using iget
            attempts: int (default: 1)
                Number of retries, if iRODS access fails
            pause: int (default: 15)
                Pause between two access attempts in seconds
        """
        if attempts > 1:
            cmd = """   for i in {{1..{0}}}; do
                            ret=$(iget -v {1} 2>&1)
                            echo $ret
                            if [[ $ret == *"ERROR"* ]]; then
                                echo "Attempt $i failed"
                            else
                                break
                            fi
                            sleep {2}s
                        done """
            cmd = lstrip(cmd)
            cmd = cmd.format(attempts, irods_path, pause)
            self.add(cmd)
        else:
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
        return "\n".join(self.lines)

    def __repr__(self):
        return "# Shell script\n" + str(self)
