# Filename: tohdf5.py
"""
Convert ROOT and EVT files to HDF5.

Usage:
    tohdf5 [options] FILE...
    tohdf5 (-h | --help)
    tohdf5 --version

Options:
    -h --help                       Show this screen.
    --verbose                       Print more output.
    --debug                         Print everything.
    -n EVENTS                       Number of events/runs.
    -o OUTFILE                      Output file.
    -j --jppy                       (Jpp): Use jppy (not aanet) for Jpp readout.
    --ignore-hits                   Don't read the hits.
    -e --expected-rows NROWS        Approximate number of events.  Providing a
                                    rough estimate for this (100, 1000000, ...)
                                    will greatly improve reading/writing speed
                                    and memory usage.
                                    Strongly recommended if the table/array
                                    size is >= 100 MB. [default: 10000]
    -t --conv-times-to-jte          Converts all MC times in the file to JTE
                                    times.
"""

from km3modules.common import StatusBar
from km3pipe import version
from km3pipe import logger

__author__ = "Tamas Gal and Michael Moser"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = logger.get_logger('km3pipe.io')


def tohdf5(input_files, output_file, n_events, conv_times_to_jte,
           **kwargs):
    """Convert Any file to HDF5 file"""
    from km3pipe import Pipeline    # noqa
    from km3pipe.io import GenericPump, HDF5Sink, HDF5MetaData    # noqa

    pipe = Pipeline()
    pipe.attach(GenericPump, filenames=input_files, **kwargs)
    pipe.attach(HDF5MetaData, data=kwargs)
    pipe.attach(StatusBar, every=250)
    if conv_times_to_jte: 
        from km3modules.mc import MCTimeCorrector
        pipe.attach(MCTimeCorrector)
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
    if len(infiles) == 1:
        suffix = '.h5'
    else:
        # if the user is too lazy specifying an outfile name
        # when converting *multiple files into one* (yeah, I know),
        # at least be clear that it's a combined file
        suffix = '.combined.h5'
    outfile = args['-o'] or infiles[0] + suffix

    n_rows_expected = int(args['--expected-rows'])
    use_jppy_pump = args['--jppy']
    is_verbose = bool(args['--verbose'])
    if is_verbose:
        log.setLevel('INFO')
    is_debug = bool(args['--debug'])
    if is_debug:
        log.setLevel('DEBUG')
    ignore_hits_arg = args['--ignore-hits']
    conv_times_to_jte = bool(args['--conv-times-to-jte'])
    tohdf5(
        infiles,
        outfile,
        n,
        conv_times_to_jte,
        use_jppy=use_jppy_pump,
        n_rows_expected=n_rows_expected,
        ignore_hits=bool(ignore_hits_arg),
    )
