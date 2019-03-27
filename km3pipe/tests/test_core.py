# Filename: test_core.py
# pylint: disable=C0111,E1003,R0904,C0103,R0201,C0102
from __future__ import unicode_literals

import tempfile
from io import StringIO

from km3pipe.testing import TestCase, MagicMock, patch
from km3pipe.core import Pipeline, Module, Pump, Blob

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestPump(TestCase):
    """Tests for the pump"""

    def test_rewind_file(self):
        pump = Pump()
        test_file = StringIO("Some content")
        pump.blob_file = test_file
        pump.blob_file.read(1)
        self.assertEqual(1, pump.blob_file.tell())
        pump.rewind_file()
        self.assertEqual(0, pump.blob_file.tell())

    def test_context(self):
        with Pump() as p:
            print(p)
