#!/usr/bin/env python
"""Convert an I3 file to ROOT.

Direct conversion to HDF5 is possible in theory, but the Lyon seatray
installation has a broken h5/pytables install.

Usage:
    i3toroot.py INFILE
    i3toroot.py -h | --help


Options:
    -h --help     Show this screen.
"""

from docopt import docopt

# order of these imports is crucial!!!
from icecube import icetray, dataio    # noqa
from I3Tray import I3Tray
from icecube.tableio import I3TableWriter
from icecube.rootwriter import I3ROOTTableService


def i3toroot(infile):
    rootfile = infile + '.root'
    tray = I3Tray()
    tray.AddModule('I3Reader', 'i3reader', filename=infile)
    root = I3ROOTTableService(rootfile)
    tray.AddModule(
        I3TableWriter,
        'writer',
        tableservice=root,
        BookEverything=True,
    )
    tray.AddModule('TrashCan', 'dustbin')
    tray.Execute()
    tray.Finish()


def main():
    args = docopt(__doc__)
    infile = args['INFILE']
    i3toroot(infile)


if __name__ == '__main__':
    main()
