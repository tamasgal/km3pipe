# coding=utf-8
# Filename: rba.py
"""
RainbowAlga online display.

Usage:
    rba -t TOKEN -n EVENT_ID [-u URL] FILE
    rba (-h | --help)
    rba --version

Options:
    FILE       Input file.
    -h --help  Show this screen.

"""
from __future__ import division, absolute_import, print_function

import tables


def rba():
    pass


def main():
    from docopt import docopt
    arguments = docopt(__doc__)

    rba(arguments['FILE'])
