#!/usr/bin/env python3
# Filename: summaryex.py
"""
Extract information from multiple ROOT files and write a summary file.

Usage:
    summaryex [options] -o OUTFILE FILES...
    summaryex (-h | --help)
    summaryex --version

Options:
    -h --help                       Show this screen.
    --verbose                       Print more output.
    --debug                         Print everything.
    -o OUTFILE                      Output file (only if one file is converted).
"""
import km3pipe as kp
import km3modules as km
from docopt import docopt


class FilesPump(kp.Module):
    pass


def main():
    args = docopt(__doc__)
    print(args)
    is_verbose = args['--verbose']
    is_debug = args['--debug']

    pipe = kp.Pipeline(timeit=is_verbose or is_debug)
    pipe.attach(FilesPump)
