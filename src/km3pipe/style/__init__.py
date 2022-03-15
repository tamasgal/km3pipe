# Filename: style.py
# pylint: disable=locally-disabled
"""
The KM3Pipe style definitions.

"""

from os.path import dirname, join, exists
from itertools import cycle

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

STYLE_DIR = join(dirname(dirname(__file__)), "stylelib")


def get_style_path(style):
    return STYLE_DIR + "/" + style + ".mplstyle"


def use(style="km3pipe"):
    import matplotlib.pyplot as plt

    for s in (get_style_path("km3pipe-" + style), get_style_path(style), style):
        if exists(s):
            plt.style.use(s)
            return
    print("Could not find style: '{0}'".format(style))


class ColourCycler(object):
    """Basic colour cycler.

    Instantiate with `cc = ColourCycler()` and use it in plots
    like `plt.plot(xs, ys, c=next(cc))`.
    """

    def __init__(self, palette="km3pipe"):
        self.colours = {}
        self.refresh_styles()
        self.choose(palette)

    def choose(self, palette):
        """Pick a palette"""
        try:
            self._cycler = cycle(self.colours[palette])
        except KeyError:
            raise KeyError(
                "Chose one of the following colour palettes: {0}".format(self.available)
            )

    def refresh_styles(self):
        """Load all available styles"""
        import matplotlib.pyplot as plt

        self.colours = {}
        for style in plt.style.available:
            try:
                style_colours = plt.style.library[style]["axes.prop_cycle"]
                self.colours[style] = [c["color"] for c in list(style_colours)]
            except KeyError:
                continue

        self.colours["km3pipe"] = [
            "#ff7869",
            "#4babe1",
            "#96ad3e",
            "#e4823d",
            "#5d72b2",
            "#e2a3c2",
            "#fd9844",
            "#e480e7",
        ]

    @property
    def available(self):
        """Return a list of available styles"""
        return list(self.colours.keys())

    def __next__(self):
        """Return the next colour in current palette"""
        return next(self._cycler)

    def next(self):
        """Python 2 compatibility for iterators"""
        return self.__next__()
