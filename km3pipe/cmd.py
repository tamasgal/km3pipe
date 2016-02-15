# coding=utf-8
# Filename: cmd.py
"""
KM3Pipe command line utility.

Usage:
    km3pipe test
    km3pipe (-h | --help)
    km3pipe --version

Options:
    -h --help       Show this screen.

"""
from __future__ import division, absolute_import, print_function

from km3pipe import version


def main():
    from docopt import docopt
    arguments = docopt(__doc__, version=version)
    print(arguments)
