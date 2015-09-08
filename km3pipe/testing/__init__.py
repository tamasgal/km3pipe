# coding=utf-8
# Filename: __init__.py
"""
Common unit testing support for km3pipe.

"""
from __future__ import division, absolute_import, print_function

try:
    from unittest2 import TestCase, skip, skipIf
except ImportError:
    from unittest import TestCase, skip, skipIf

try:
    from cStringIO import StringIO
except ImportError:
    try:
        from StringIO import StringIO
    except ImportError:
        from io import StringIO

from mock import MagicMock
