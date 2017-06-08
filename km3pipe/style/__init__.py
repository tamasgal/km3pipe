# coding=utf-8
# Filename: style.py
# pylint: disable=locally-disabled
"""
The KM3Pipe style definitions.

"""
from __future__ import division, absolute_import, print_function

import os
from itertools import cycle

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("Please install matplotlib: `pip install matplotlib`")
import km3pipe as kp


__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


style_dir = os.path.dirname(kp.__file__) + '/kp-data/stylelib'


class KM3PipePygmentsStyle(object):
    def __init__(self):
        raise ImportError("Module pygments could be not found")

class LatexPreprocessor(object):
    def __init__(self):
        raise ImportError("Module nbconvert could be not found")

try:
    from nbconvert.preprocessors.base import Preprocessor
    from pygments.style import Style
    import pygments.token as token
except ImportError:
    pass
else:
    class KM3PipePygmentsStyle(Style):
        default_style = ""
        styles = {
            token.Comment:         '#aaa',
            token.Keyword:         '#0095B5',
            token.Keyword.Control: '#0095B5',
            token.Name:            '#333',
            token.Name.Function:   '#FF0082',
            token.Name.Class:      'bold #FF0082',
            token.String:          '#666',
            token.Operator:        '#346F8A',
        }

    class LatexPreprocessor(Preprocessor):
        """LaTeX processor for nbconvert"""
        def preprocess(self, nb, resources):
            from pygments.formatters import LatexFormatter
            resources["latex"]["pygments_definitions"] = \
                    LatexFormatter(style=KM3PipePygmentsStyle).get_style_defs()
            return nb, resources


def get_style_path(style):
    return style_dir + '/' + style + '.mplstyle'


def use(style='km3pipe'):
    for s in (get_style_path('km3pipe-'+style),
              get_style_path(style),
              style):
        try:
            plt.style.use(s)
        except (OSError, IOError):
            pass
        else:
            print("Loading style definitions from '{0}'".format(s))
            return
    print("Could not find style: '{0}'".format(style))


class ColourCycler(object):
    """Basic colour cycler.

    Instantiate with `cc = ColourCycler()` and use it in plots
    like `plt.plot(xs, ys, c=cs.next)`.
    """
    def __init__(self, palette='km3pipe'):
        self.colours = {}
        self.refresh_styles()
        self.choose(palette)

    def choose(self, palette):
        """Pick a palette"""
        try:
            self._cycler = cycle(self.colours[palette])
        except KeyError:
            raise KeyError("Chose one of the following colour palettes: {0}"
                           .format(self.available))

    def refresh_styles(self):
        """Load all available styles"""
        self.colours = {}
        for style in plt.style.available:
            try:
                style_colours = plt.style.library[style]['axes.prop_cycle']
                self.colours[style] = [c['color'] for c in list(style_colours)]
            except KeyError:
                continue

        self.colours['km3pipe'] = ["#ff7869", "#4babe1", "#96ad3e",
                                   "#e4823d", "#5d72b2", "#e2a3c2",
                                   "#fd9844", "#e480e7"]

    @property
    def available(self):
        """Return a list of available styles"""
        return list(self.colours.keys())

    @property
    def next(self):
        """Return the next colour in current palette"""
        return next(self._cycler)
