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
    from mock import Mock
    from mock import patch
except ImportError:
    from unittest.mock import MagicMock  # noqa
    from unittest.mock import Mock  # noqa
    from unittest.mock import patch  # noqa

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"
