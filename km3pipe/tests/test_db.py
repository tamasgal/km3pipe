# Filename: test_db.py
# pylint: disable=C0111,E1003,R0904,C0103,R0201,C0102
from os.path import dirname, join
from km3pipe.testing import TestCase, MagicMock, patch

from km3pipe.db import (
    DBManager, DOMContainer, we_are_in_lyon, read_csv, make_empty_dataset,
    StreamDS
)
from km3pipe.logger import get_logger

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

DET_ID = 'det_id1'
JSON_DOMS = [{
    'DOMId': 1,
    'Floor': 10,
    'CLBUPI': '100',
    'DetOID': DET_ID
},
             {
                 'DOMId': 2,
                 'Floor': 20,
                 'CLBUPI': '200',
                 'DetOID': DET_ID
             },
             {
                 'DOMId': 3,
                 'Floor': 30,
                 'CLBUPI': '300',
                 'DetOID': DET_ID
             }, {
                 'DOMId': 4,
                 'Floor': 40,
                 'CLBUPI': '400',
                 'DetOID': 'det_id2'
             }]

log = get_logger('db')

STREAMDS_META = join(
    dirname(__file__), "../kp-data/test_data/streamds_output.txt"
)


class TestDBManager(TestCase):
    def test_login_called_on_init_when_credentials_are_provided(self):
        user = 'user'
        pwd = 'god'

        DBManager.login = MagicMock()
        DBManager(username=user, password=pwd, temporary=True)
        self.assertEqual(1, DBManager.login.call_count)
        self.assertTupleEqual((user, pwd), DBManager.login.call_args[0])

    def test_login(self):
        original_login = DBManager.login    # save for later

        # mock login to be able to create an instance without an actual login
        DBManager.login = MagicMock()
        db = DBManager(username='foo', password='bar')    # make dummy call
        DBManager.login = original_login    # restore function

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


class TestWeAreInLyon(TestCase):
    @patch('socket.gethostbyname')
    @patch('socket.gethostname')
    def test_call_in_lyon(self, gethostname_mock, gethostbyname_mock):
        gethostbyname_mock.return_value = '134.158.'
        assert we_are_in_lyon()

    @patch('socket.gethostbyname')
    @patch('socket.gethostname')
    def test_call_not_in_lyon(self, gethostname_mock, gethostbyname_mock):
        gethostbyname_mock.return_value = '1.2.'
        assert not we_are_in_lyon()


class TestDataSetFunctions(TestCase):
    def test_read_csv(self):
        raw = "a\tb\n1\t2\n3\t4"
        df = read_csv(raw)
        assert len(df) == 2
        self.assertListEqual([1, 3], list(df.a))
        self.assertListEqual([2, 4], list(df.b))

    def test_make_empty_dataset(self):
        df = make_empty_dataset()
        assert len(df) == 0


class TestStreamDS(TestCase):
    @patch('km3pipe.db.DBManager._get_content')
    @patch('km3pipe.db.DBManager')
    def setUp(self, db_manager_mock, get_content_mock):
        with open(STREAMDS_META, 'r') as fobj:
            streamds_meta = fobj.read()
        db_manager_mock_obj = db_manager_mock.return_value
        db_manager_mock_obj._get_content.return_value = streamds_meta
        self.sds = StreamDS()

    def test_streams(self):
        assert len(self.sds.streams) == 30
        assert 'ahrs' in self.sds.streams
        assert 'clbmap' in self.sds.streams
        assert 'datalogevents' in self.sds.streams
        assert 'datalognumbers' in self.sds.streams

    def test_getattr(self):
        assert hasattr(self.sds, 't0sets')
        assert hasattr(self.sds, 'vendorhv')

    def test_attr_are_callable(self):
        self.sds.runs()
        self.sds.runsetupparams()
        self.sds.upi()

    def test_mandatory_selectors(self):
        self.assertListEqual(['-'], self.sds.mandatory_selectors('productloc'))
        self.assertListEqual(['detid'], self.sds.mandatory_selectors('runs'))
        self.assertListEqual(['detid', 'minrun', 'maxrun'],
                             self.sds.mandatory_selectors('datalognumbers'))

    def test_optional_selectors(self):
        self.assertListEqual([
            'upi', 'city', 'locationid', 'operation', 'operationid'
        ], self.sds.optional_selectors('productloc'))
        self.assertListEqual(['run', 'runjobid', 'jobtarget', 'jobpriority'],
                             self.sds.optional_selectors('runs'))
        self.assertListEqual(['source_name', 'parameter_name'],
                             self.sds.optional_selectors('datalognumbers'))

    def test_print_streams(self):
        self.sds.print_streams()
