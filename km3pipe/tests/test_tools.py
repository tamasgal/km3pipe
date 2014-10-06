from __future__ import division, absolute_import, print_function

from km3pipe.testing import *
from km3pipe.tools import unpack_nfirst


class TestTools(TestCase):

    def test_unpack_nfirst(self):
        a_tuple = (1, 2, 3, 4, 5)
        a, b, c, rest = unpack_nfirst(a_tuple, 3)
        self.assertEqual(1, a)
        self.assertEqual(2, b)
        self.assertEqual(3, c)
        self.assertTupleEqual((4, 5), rest)