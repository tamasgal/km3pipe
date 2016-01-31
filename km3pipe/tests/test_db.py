# coding=utf-8
# Filename: test_core.py
# pylint: disable=C0111,E1003,R0904,C0103,R0201,C0102
from __future__ import division, absolute_import, print_function

from km3pipe.testing import TestCase

from km3pipe.db import DOMContainer

__author__ = 'tamasgal'


DET_ID = 'det_id1'
JSON_DOMS = [{'DOMId': 1, 'Floor': 10, 'CLBUPI': '100', 'DetOID': DET_ID},
             {'DOMId': 2, 'Floor': 20, 'CLBUPI': '200', 'DetOID': DET_ID},
             {'DOMId': 3, 'Floor': 30, 'CLBUPI': '300', 'DetOID': DET_ID},
             {'DOMId': 4, 'Floor': 40, 'CLBUPI': '400', 'DetOID': 'det_id2'}]


class TestDBManager(TestCase):
    pass


class TestDOMContainer(TestCase):
    def test_init(self):
        DOMContainer(None)

    def setUp(self):
        self.dc = DOMContainer(JSON_DOMS)

    def test_ids_returns_dom_ids(self):
        self.assertListEqual([1, 2, 3], self.dc.ids(DET_ID))

    def test_json_list_lookup(self):
        lookup = self.dc._json_list_lookup('DOMId', 1, 'Floor', DET_ID)
        self.assertEqual(10, lookup)

    def test_clbupi2floor(self):
        self.assertEqual(10, self.dc.clbupi2floor('100', DET_ID))
        self.assertEqual(20, self.dc.clbupi2floor('200', DET_ID))
        self.assertEqual(30, self.dc.clbupi2floor('300', DET_ID))

    def test_clbupi2domid(self):
        self.assertEqual(1, self.dc.clbupi2domid('100', DET_ID))
        self.assertEqual(2, self.dc.clbupi2domid('200', DET_ID))
        self.assertEqual(3, self.dc.clbupi2domid('300', DET_ID))
