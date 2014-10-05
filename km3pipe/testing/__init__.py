from __future__ import division, absolute_import, print_function

try:
    from unittest2 import TestCase
except ImportError:
    from unittest import TestCase

try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

from mock import MagicMock