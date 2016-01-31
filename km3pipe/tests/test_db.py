# coding=utf-8
# Filename: test_core.py
# pylint: disable=C0111,E1003,R0904,C0103,R0201,C0102
from __future__ import division, absolute_import, print_function

from km3pipe.testing import TestCase

from km3pipe.db import DOMContainer

__author__ = 'tamasgal'


class TestDBManager(TestCase):
    pass


class TestDOMContainer(TestCase):
    def test_init(self):
        DOMContainer(None)

    def test_ids_returns_dom_ids(self):
        det_id = 'a'
        json_data = [{'DOMId': 1, 'DetOID': det_id},
                     {'DOMId': 2, 'DetOID': det_id},
                     {'DOMId': 3, 'DetOID': det_id},
                     {'DOMId': 4, 'DetOID': 'another_det_id'}]
        dc = DOMContainer(json_data)
        self.assertListEqual([1, 2, 3], dc.ids(det_id))
