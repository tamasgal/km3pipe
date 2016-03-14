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
