# Filename: test_foo.py
# pylint: disable=C0111,E1003,R0904,C0103,R0201,C0102
"""Make a dummy test so jenkins won't cry ):
"""
from km3pipe.testing import TestCase

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2016,  and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestPipeline(TestCase):
    """Tests for the main pipeline"""

    def setUp(self):
        pass

    def test_pass(self):
        assert True
