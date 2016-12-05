# coding=utf-8
# Filename: style.py
# pylint: disable=locally-disabled
"""
The KM3Pipe style definitions.

"""
from __future__ import division, absolute_import, print_function

import os

import matplotlib.pyplot as plt
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


def get_style_path(style='km3pipe'):
    if style in ('default', 'km3pipe'):
        style = ''
    else:
        style = '-' + style
    return style_dir + '/km3pipe' + style + '.mplstyle'


def use(style):
    if style in ('default', 'km3pipe', 'talk', 'notepad', 'poster'):
        style = get_style_path(style)
        plt.style.use(get_style_path('default'))
    plt.style.use(style)


# Automatically load default style on import.
use('default')
