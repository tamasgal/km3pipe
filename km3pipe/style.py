# coding=utf-8
# Filename: style.py
# pylint: disable=locally-disabled
"""
The KM3Pipe style definitions.

"""
from __future__ import division, absolute_import, print_function

import os

import matplotlib.pyplot as plt
import matplotlib.style
import km3pipe as kp

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


style_dir = os.path.dirname(kp.__file__) + '/kp-data/stylelib'


def get_style_path(style='km3pipe'):
    if style in ('default', 'km3pipe'):
        style = ''
    else:
        style = '-' + style
    return style_dir + '/km3pipe' + style + '.mplstyle'


def use(style):
    if style not in matplotlib.style.available:
        style = get_style_path(style)
    plt.style.use(style)


# Automatically load default style on import.
use('default')
