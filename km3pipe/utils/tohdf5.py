# coding=utf-8
# Filename: tohdf5.py
"""
Convert ROOT and EVT files to HDF5.

Usage:
    tohdf5 [options] FILE...
    tohdf5 (-h | --help)
    tohdf5 --version

Options:
    -n EVENTS                       Number of events/runs.
    -o OUTFILE                      Output file.
    --aa-format=<fmt>               (Aanet): Which aanet subformat ('minidst',
                                    'orca_recolns', 'gandalf',
                                    'generic_track') [default: None]
    --aa-lib=<lib.so>               (Aanet): path to aanet binary (for old
                                    versions which must be loaded via
                                    `ROOT.gSystem.Load()` instead of `import aa`)
    --aa-old-mc-id                  (aanet): read mc id as `evt.mc_id`, instead
                                    of the newer `mc_id = evt.frame_index - 1`
    -h --help                       Show this screen.
    -j --jppy                       (Jpp): Use jppy (not aanet) for Jpp readout
    -l --with-timeslice-hits        Include timeslice-hits [default: False]
    -s --with-summaryslices         Include summary slices [default: False]
    --correct-zed                   (Aanet) Correct offset in mc tracks (aanet)
                                    [default: False]
    --do-not-correct-mc-times       Don't correct MC times.
    --skip-header                   (Aanet) don't read the full header.
                                    Entries like `genvol` and `neventgen` will
                                    still be retrived. This switch enables
                                    skipping the `get_aanet_header` function only.
                                    [default: False]
    -e --expected-rows NROWS        Approximate number of events.  Providing a
                                    rough estimate for this (100, 10000000, ...)
                                    will greatly improve reading/writing speed and
                                    memory usage. Strongly recommended if the
                                    table/array size is >= 100 MB. [default: 10000]
"""

from __future__ import division, absolute_import, print_function

from km3modules.common import StatusBar
from km3pipe import version

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def tohdf5(input_files, output_file, n_events, **kwargs):
    """Convert Any file to HDF5 file"""
    from km3pipe import Pipeline  # noqa
    from km3pipe.io import GenericPump, HDF5Sink  # noqa

    pipe = Pipeline()
    pipe.attach(GenericPump, filenames=input_files, **kwargs)
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
    aa_format = args['--aa-format']
    aa_lib = args['--aa-lib']
    aa_old_mc_id = args['--aa-old-mc-id']
    with_summaryslices = args['--with-summaryslices']
    with_timeslice_hits = args['--with-timeslice-hits']
    correct_zed = args['--correct-zed']
    skip_header = args['--skip-header']
    do_not_correct_mc_times = args['--do-not-correct-mc-times']
    tohdf5(infiles,
           outfile,
           n,
           use_jppy=use_jppy_pump,
           aa_fmt=aa_format,
           aa_lib=aa_lib,
           with_summaryslices=with_summaryslices,
           with_timeslice_hits=with_timeslice_hits,
           n_rows_expected=n_rows_expected,
           apply_zed_correction=correct_zed,
           old_mc_id=aa_old_mc_id,
           skip_header=skip_header,
           correct_mc_times=not bool(do_not_correct_mc_times)
           )
