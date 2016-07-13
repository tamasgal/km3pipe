# coding=utf-8
# Filename: common.py
# pylint: disable=locally-disabled
"""
Commonly used imports.

"""
from __future__ import division, absolute_import, print_function

try:
    from cStringIO import StringIO
except ImportError:
    try:
        from StringIO import StringIO
    except ImportError:
        from io import StringIO

try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty


__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"
