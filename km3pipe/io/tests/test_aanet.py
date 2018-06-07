# Filename: test_aanet.py
# pylint: disable=locally-disabled,C0111,R0904,C0301,C0103,W0212
from km3pipe.testing import TestCase, patch, Mock, skip
from km3pipe.io.aanet import AanetPump

import sys
sys.modules['ROOT'] = Mock()
sys.modules['aa'] = Mock()

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2018, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestAanetPump(TestCase):
    def test_init_raises_valueerror_if_no_filename_given(self):
        with self.assertRaises(TypeError):
            AanetPump()

    @skip
    def test_init_with_filename(self):
        filename = 'a'
        p = AanetPump(filename=filename)
        assert filename in p.filenames
