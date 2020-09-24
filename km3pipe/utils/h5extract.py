#!/usr/bin/env python3
# Filename: h5extract.py
"""
A tool to extract data from ROOT or EVT files to HDF5.

Usage:
    h5extract filename
    h5extract (-h | --help)
    h5extract --version

Options:
    -h --help           Show this screen.
    --version           Show the version.

"""
import km3pipe as kp

log = kp.logger.get_logger(__name__)




def main():
    from docopt import docopt

    args = docopt(__doc__, version=kp.version)

    print(args)
