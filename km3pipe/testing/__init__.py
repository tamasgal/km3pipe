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


class FakeVec(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class FakeAanetHit(object):
    def __init__(self, channel_id, dir_x, dir_y, dir_z, dom_id, id, pmt_id,
                 pos_x, pos_y, pos_z, t0, t, tot, trig):
        # self.channel_id = chr(channel_id)
        self.channel_id = channel_id
        self.dir = FakeVec(dir_x, dir_y, dir_z)
        self.dom_id = dom_id
        self.id = id
        self.pmt_id = pmt_id
        self.pos = FakeVec(pos_x, pos_y, pos_z)
        self.t0 = t0
        self.t = t
        self.tot = tot
        self.trig = trig
