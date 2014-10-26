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

    def test_scale_labels(self):
        blobs = BlobWidget()
