# Filename: test_widgets.py
"""
...

"""

from km3pipe.testing import TestCase, skip

from pipeinspector.widgets import BlobWidget

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestBlobWidget(TestCase):
    def test_make_scale_labels(self):
        blobs = BlobWidget()
        blobs.width = 25
        scale_labels = blobs._make_scale_labels(0)
        self.assertEqual("0         10        20   ", scale_labels)
        scale_labels = blobs._make_scale_labels(2)
        self.assertEqual("0         10        20   ", scale_labels)
        scale_labels = blobs._make_scale_labels(10)
        self.assertEqual("0         10        20   ", scale_labels)
        scale_labels = blobs._make_scale_labels(11)
        self.assertEqual("         10        20    ", scale_labels)
        scale_labels = blobs._make_scale_labels(4589)
        self.assertEqual(" 4580      4590      ", scale_labels)

    @skip
    def test_make_ruler(self):
        blobs = BlobWidget()
        blobs.width = 25
        ruler = blobs._make_ruler(0)
        self.assertEqual("|    '    |    '    |    ", ruler)
        ruler = blobs._make_ruler(2)
        self.assertEqual("|    '    |    '    |    ", ruler)
        ruler = blobs._make_ruler(10)
        self.assertEqual("|    '    |    '    |    ", ruler)
        ruler = blobs._make_ruler(11)
        self.assertEqual("    '    |    '    |    '", ruler)
        ruler = blobs._make_ruler(12)
        self.assertEqual("   '    |    '    |    ' ", ruler)
        ruler = blobs._make_ruler(19)
        self.assertEqual(" |    '    |    '    |   ", ruler)
        ruler = blobs._make_ruler(20)
        self.assertEqual("|    '    |    '    |    ", ruler)
        ruler = blobs._make_ruler(23)
        self.assertEqual("  '    |    '    |    '  ", ruler)
        ruler = blobs._make_ruler(109)
        self.assertEqual(" |    '    |    '    |   ", ruler)
