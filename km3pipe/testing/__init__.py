# coding=utf-8
# Filename: __init__.py
"""
Common unit testing support for km3pipe.

"""
from __future__ import division, absolute_import, print_function

try:
    from unittest2 import TestCase, skipIf
except ImportError:
    from unittest import TestCase, skipIf

try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

from mock import MagicMock
