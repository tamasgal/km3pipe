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
except ImportError:
    from unittest.mock import MagicMock  # noqa
    from unittest.mock import Mock  # noqa

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class FakeAanetHit(object):
    def __init__(self, channel_id, dom_id, id, pmt_id, t, tot, trig):
        # self.channel_id = chr(channel_id)
        self.channel_id = channel_id
        self.dom_id = dom_id
        self.id = id
        self.pmt_id = pmt_id
        self.t = t
        self.tot = tot
        self.trig = trig
