# coding=utf-8
# Filename: test_aanet.py
"""
...

"""
from __future__ import division, absolute_import, print_function

from km3pipe.testing import *

from km3pipe.pumps.aanet import AanetPump

try:
    import aa
except ImportError:
    NO_AA = True
else:
    NO_AA = False


@skipIf(NO_AA, "Skipping tests for aanet")
class TestAanetPump(TestCase):

    def test_aanetpump_init(self):
        pump = AanetPump()

