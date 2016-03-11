# coding=utf-8
# Filename: __init__.py
"""
Common unit testing support for km3pipe.

"""
from __future__ import division, absolute_import, print_function

from km3pipe.common import StringIO  # noqa
from io import BytesIO  # noqa

try:
    from unittest2 import TestCase, skip, skipIf
except ImportError:
    from unittest import TestCase, skip, skipIf  # noqa

try:
    from mock import MagicMock
except ImportError:
    from unittest.mock import MagicMock  # noqa
