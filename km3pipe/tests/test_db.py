# coding=utf-8
# Filename: test_db.py
# pylint: disable=C0111,E1003,R0904,C0103,R0201,C0102
from __future__ import division, absolute_import, print_function

from km3pipe.testing import TestCase, MagicMock

from km3pipe.db import DBManager, DOMContainer
from km3pipe.logger import logging

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

DET_ID = 'det_id1'
JSON_DOMS = [{'DOMId': 1, 'Floor': 10, 'CLBUPI': '100', 'DetOID': DET_ID},
             {'DOMId': 2, 'Floor': 20, 'CLBUPI': '200', 'DetOID': DET_ID},
             {'DOMId': 3, 'Floor': 30, 'CLBUPI': '300', 'DetOID': DET_ID},
             {'DOMId': 4, 'Floor': 40, 'CLBUPI': '400', 'DetOID': 'det_id2'}]

log = logging.getLogger('db')


class TestDBManager(TestCase):

    def test_login_called_on_init_when_credentials_are_provided(self):
        user = 'user'
        pwd = 'god'

        DBManager.login = MagicMock()
        db = DBManager(username=user, password=pwd)
        self.assertEqual(1, DBManager.login.call_count)
        self.assertTupleEqual((user, pwd), DBManager.login.call_args[0])

    def test_login(self):
        original_login = DBManager.login  # save for later
        user = 'a'
        pwd = 'b'

        # mock login to be able to create an instance without an actual login
        DBManager.login = MagicMock()
        db = DBManager(username='foo', password='bar')  # make dummy call
        DBManager.login = original_login  # restore function

        db._make_request = MagicMock()
        db.login(username='a', password='b')
        call_args = db._make_request.call_args[0]
        self.assertEqual(db._login_url, call_args[0])
        self.assertDictEqual({'usr': 'a', 'pwd': 'b'}, call_args[1])


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
