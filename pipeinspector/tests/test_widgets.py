# coding=utf-8
# Filename: test_widgets.py
"""
...

"""
from __future__ import division, absolute_import, print_function

from km3pipe.testing import *

from pipeinspector.widgets import BlobWidget

__author__ = 'tamasgal'


class TestBlobWidget(TestCase):

    def test_make_scale_labels(self):
        blobs = BlobWidget()
        scale_labels = blobs._make_scale_labels(0)
        self.assertEqual("0         10        20   ", scale_labels)
        scale_labels = blobs._make_scale_labels(1)
        self.assertEqual("         10        20    ", scale_labels)
        scale_labels = blobs._make_scale_labels(4589)
        self.assertEqual(" 4590      4600      ", scale_labels)

    def test_make_ruler(self):
        blobs = BlobWidget()
        ruler = blobs._make_ruler(0)
        self.assertEqual("|    '    |    '    |    ", ruler)
        ruler = blobs._make_ruler(1)
        self.assertEqual("    '    |    '    |    '", ruler)
        ruler = blobs._make_ruler(2)
        self.assertEqual("   '    |    '    |    ' ", ruler)
        ruler = blobs._make_ruler(9)
        self.assertEqual(" |    '    |    '    |   ", ruler)
        ruler = blobs._make_ruler(10)
        self.assertEqual("|    '    |    '    |    ", ruler)
        ruler = blobs._make_ruler(23)
        self.assertEqual("  '    |    '    |    '  ", ruler)
        ruler = blobs._make_ruler(109)
        self.assertEqual(" |    '    |    '    |   ", ruler)