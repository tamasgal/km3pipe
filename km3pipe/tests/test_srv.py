# Filename: test_srv.py
# pylint: disable=locally-disabled,C0111,R0904,C0103
from km3pipe.testing import TestCase, patch
from km3pipe.dataclasses import Table
from km3pipe.srv import srv_event

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestSrvEvent(TestCase):
    @patch('km3pipe.srv.srv_data')
    def test_call(self, srv_data_mock):
        hits = Table({
            'pos_x': [1, 2],
            'pos_y': [3, 4],
            'pos_z': [5, 6],
            'time': [100, 200],
            'tot': [11, 22]
        })
        srv_event('token', hits, 'rba_url')
        srv_data_mock.assert_called_once()
