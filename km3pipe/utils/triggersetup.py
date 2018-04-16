# Filename: triggersetup.py
"""
Prints the trigger information of a given run setup.

Usage:
    triggersetup RUNSETUP_OID
    triggersetup (-h | --help)
    triggersetup --version

Options:
    RUNSETUP_OID   The run setup identifier (e.g. A02004580)
    -h --help      Show this screen.

"""

import km3pipe as kp

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2017, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def main():
    from docopt import docopt
    args = docopt(__doc__, version=kp.version)

    db = kp.db.DBManager()
    print(db.trigger_setup(args['RUNSETUP_OID']))
