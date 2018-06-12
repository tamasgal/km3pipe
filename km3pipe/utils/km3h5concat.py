# Filename: km3h5concat.py
"""
Concatenate KM3HDF5 files via pipeline.

Usage:
    km3h5concat [options] OUTFILE FILE...
    km3h5concat (-h | --help)
    km3h5concat --version

Options:
    -h --help                       Show this screen.
    --verbose                       Print more output.
    --debug                         Print everything.
    -n=NEVENTS                      Number of events; if not given, use all.
"""

from km3modules.common import StatusBar
from km3pipe import version
from km3pipe import logger

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"

log = logger.get_logger('km3pipe.io')


def km3h5concat(input_files, output_file, n_events=None, **kwargs):
    """Concatenate KM3HDF5 files via pipeline."""
    from km3pipe import Pipeline    # noqa
    from km3pipe.io import HDF5Pump, HDF5Sink    # noqa

    pipe = Pipeline()
    pipe.attach(HDF5Pump, filenames=input_files, **kwargs)
    pipe.attach(StatusBar, every=250)
    pipe.attach(HDF5Sink, filename=output_file, **kwargs)
    pipe.drain(n_events)


def main():
    from docopt import docopt
    args = docopt(__doc__, version=version)

    try:
        n = int(args['-n'])
    except TypeError:
        n = None

    # FILE... (ellipsis) always returns a list in docopt.
    # so the bug checking-string-length should not happen here
    infiles = args['FILE']
    outfile = args['OUTFILE']

    is_verbose = bool(args['--verbose'])
    if is_verbose:
        log.setLevel('INFO')
    is_debug = bool(args['--debug'])
    if is_debug:
        log.setLevel('DEBUG')
    km3h5concat(infiles, outfile, n)
