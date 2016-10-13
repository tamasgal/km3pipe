# coding=utf-8
# Filename: test_style.py
# pylint: disable=C0111,E1003,R0904,C0103,R0201,C0102
from __future__ import division, absolute_import, print_function

from km3pipe.testing import TestCase
from km3pipe.style import get_style_path

import matplotlib.pyplot as plt
import os

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestStyle(TestCase):

    def test_get_style_path(self):
        gsp = get_style_path
        default = "kp-data/stylelib/km3pipe.mplstyle"
        self.assertTrue(gsp().startswith('/'))
        self.assertTrue(gsp().endswith(default))
        self.assertTrue(gsp('km3pipe').endswith(default))
        self.assertTrue(gsp('default').endswith(default))
        self.assertTrue(gsp('foo').endswith("/stylelib/km3pipe-foo.mplstyle"))
        self.assertTrue(gsp('bar').endswith("/stylelib/km3pipe-bar.mplstyle"))
