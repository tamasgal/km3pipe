#!/usr/bin/env python
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
    -o OUTFILE                      Output file (only if one file is converted).
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

log = logger.get_logger('tohdf5')
cprint = logger.get_printer('tohdf5')
io_log = logger.get_logger('km3pipe.io')


def tohdf5(input_files, output_file, n_events, conv_times_to_jte, **kwargs):
    """Convert Any file to HDF5 file"""
    if len(input_files) > 1:
        cprint(
            "Preparing to convert {} files to HDF5.".format(len(input_files))
        )

    from km3pipe import Pipeline    # noqa
    from km3pipe.io import GenericPump, HDF5Sink, HDF5MetaData    # noqa

    for input_file in input_files:
        cprint("Converting '{}'...".format(input_file))
        if len(input_files) > 1:
            output_file = input_file + '.h5'

        meta_data = kwargs.copy()
        meta_data['origin'] = input_file

        pipe = Pipeline()
        pipe.attach(HDF5MetaData, data=meta_data)
        pipe.attach(GenericPump, filenames=input_file, **kwargs)
        pipe.attach(StatusBar, every=250)
        if conv_times_to_jte:
            from km3modules.mc import MCTimeCorrector
            pipe.attach(MCTimeCorrector)
        pipe.attach(HDF5Sink, filename=output_file, **kwargs)
        pipe.drain(n_events)
        cprint("File '{}' was converted.".format(input_file))


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
    outfile = args['-o']
    if len(infiles) == 1 and outfile is None:
        outfile = infiles[0] + '.h5'
    if len(infiles) > 1 and outfile is not None:
        log.warning("Ignoring output file name for multiple files.")
        outfile = None

    n_rows_expected = int(args['--expected-rows'])
    use_jppy_pump = args['--jppy']
    is_verbose = bool(args['--verbose'])
    if is_verbose:
        io_log.setLevel('INFO')
    is_debug = bool(args['--debug'])
    if is_debug:
        io_log.setLevel('DEBUG')
    ignore_hits_arg = args['--ignore-hits']
    conv_times_to_jte = bool(args['--conv-times-to-jte'])
    tohdf5(
        input_files=infiles,
        output_file=outfile,
        n_events=n,
        conv_times_to_jte=conv_times_to_jte,
        use_jppy=use_jppy_pump,
        n_rows_expected=n_rows_expected,
        ignore_hits=bool(ignore_hits_arg),
    )


if __name__ == '__main__':
    main()
