# Filename: app.py
"""
PipeInspector

Usage:
    pipeinspector FILE
    pipeinspector (-h | --help)
    pipeinspector --version

Options:
    -h --help       Show this screen.

"""

import os

from pipeinspector.settings import UI
import km3pipe.extras
from km3pipe.io import OfflinePump, HDF5Pump, EvtPump, CLBPump
from km3pipe.io.daq import DAQPump

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"
__version__ = "1.0.0"


def handle_input(input):
    """Handle any unhandled input."""
    if input in UI.keys["escape"]:
        raise urwid.ExitMainLoop


def filter_input(keys, raw):
    """Adds fancy mouse wheel functionality and VI navigation to ListBox"""
    if len(keys) == 1:
        if keys[0] in UI.keys["up"]:
            keys[0] = "up"
        elif keys[0] in UI.keys["down"]:
            keys[0] = "down"
        elif len(keys[0]) == 4 and keys[0][0] == "mouse press":
            if keys[0][1] == 4:
                keys[0] = "up"
            elif keys[0][1] == 5:
                keys[0] = "down"
    return keys


def get_pump(input_file):
    extension = os.path.splitext(input_file)[1][1:]
    if extension == "evt":
        pump = EvtPump(filename=input_file, cache_enabled=True)
    elif extension == "dat":
        pump = DAQPump(filename=input_file)
    elif extension == "dqd":
        pump = CLBPump(filename=input_file, cache_enabled=True)
    elif extension == "root":
        pump = OfflinePump(filename=input_file)
    elif extension == "h5":
        pump = HDF5Pump(filename=input_file)
    else:
        raise SystemExit("No pump found for '{0}' files.".format(extension))
    return pump


def main():
    from docopt import docopt

    arguments = docopt(__doc__, version=__version__)

    urwid = km3pipe.extras.urwid()

    input_file = arguments["FILE"]
    pump = get_pump(input_file)

    from pipeinspector.gui import MainFrame

    main_frame = MainFrame(pump)
    # main_frame.header.set_text("Inspecting {0}".format(input_file))

    loop = urwid.MainLoop(
        main_frame, UI.palette, input_filter=filter_input, unhandled_input=handle_input
    )
    loop.run()


if __name__ == "__main__":
    main()
