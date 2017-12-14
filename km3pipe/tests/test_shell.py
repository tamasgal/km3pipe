# coding=utf-8
# Filename: test_shell.py
# pylint: disable=locally-disabled,C0111,R0904,C0103
from __future__ import division, absolute_import, print_function

from km3pipe.testing import TestCase
from km3pipe.shell import Script

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2017, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestScript(TestCase):
    def test_add(self):
        s = Script()
        s.add("a")
        s.add("b")

    def test_str(self):
        s = Script()
        s.add("a")
        s.add("b")
        self.assertEqual("a\nb", str(s))

    def test_clear(self):
        s = Script()
        s.add("a")
        s.clear()
        self.assertEqual('', str(s))
