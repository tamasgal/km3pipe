# Filename: wtd.py
"""
Prints information for a given DOM (and detector [O]ID)

Usage:
    wtd DET_ID_OR_OID DOM_ID
    wtd (-h | --help)
    wtd --version

Options:
    DOM_ID          The actual DOM ID.
    DET_ID_OR_OID   Detector ID (like 29) or OID (like D_ARCA003).
    -h --help       Show this screen.

"""

import km3pipe as kp

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2018, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def main():
    from docopt import docopt
    args = docopt(__doc__, version=kp.version)

    db = kp.db.DBManager()

    dom_id = int(args['DOM_ID'])
    det = args['DET_ID_OR_OID']

    try:
        det = int(det)
    except ValueError:
        pass

    print(db.doms.via_dom_id(dom_id, det).__repr__())
