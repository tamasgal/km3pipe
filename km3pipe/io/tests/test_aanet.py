# Filename: test_aanet.py
# pylint: disable=locally-disabled,C0111,R0904,C0301,C0103,W0212
from os.path import join, dirname

import numpy as np

from km3pipe.testing import TestCase, patch, Mock, skip
from km3pipe.io.aanet import AanetPump, MetaParser

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

TEST_DATA_DIR = join(dirname(__file__), "../../kp-data/test_data")


class TestAanetPump(TestCase):
    def test_init_raises_valueerror_if_no_filename_given(self):
        with self.assertRaises(TypeError):
            AanetPump()

    @skip
    def test_init_with_filename(self):
        filename = 'a'
        p = AanetPump(filename=filename)
        assert filename in p.filenames


class TestMetaParser(TestCase):
    def test_init(self):
        MetaParser()

    def test_parse_string_of_single_entry(self):
        string = b"A 123\nA 1.2.3\nA KM3NET\nA a\nA b\nA Linux"
        mp = MetaParser()
        mp.parse_string(string)

        assert 1 == len(mp.meta)

        assert b'A' == mp.meta[0]['application_name']
        assert b'123' == mp.meta[0]['revision']
        assert b'A a\nA b' == mp.meta[0]['command']

    def test_parse_string_of_multiple_entries(self):
        string = (
            b"A 123\nA 1.2.3\nA KM3NET\nA a\nA b\nA Linux\n"
            b"B 456\nB 4.5.6\nB KM3NET\nB c\nB Linux"
        )
        mp = MetaParser()
        mp.parse_string(string)

        assert 2 == len(mp.meta)

        assert b'A' == mp.meta[0]['application_name']
        assert b'123' == mp.meta[0]['revision']
        assert b'1.2.3' == mp.meta[0]['root_version']
        assert b'A a\nA b' == mp.meta[0]['command']

        assert b'B' == mp.meta[1]['application_name']
        assert b'456' == mp.meta[1]['revision']
        assert b'4.5.6' == mp.meta[1]['root_version']
        assert b'B c' == mp.meta[1]['command']

    def test_parse_testfile(self):
        fname = join(TEST_DATA_DIR, 'jprintmeta.log')
        with open(fname, 'rb') as fobj:
            string = fobj.read()
        mp = MetaParser()
        mp.parse_string(string)
        assert 7 == len(mp.meta)

        assert b'JEvt' == mp.meta[0]['application_name']
        assert b'9912' == mp.meta[0]['revision']
        assert b'5.34/23' == mp.meta[0]['root_version']
        assert mp.meta[0]['command'].startswith(b'JEvt /pbs/throng/km3net')
        assert mp.meta[0]['command'].endswith(b'2 --!')

        assert b'JEnergy' == mp.meta[1]['application_name']
        assert b'9912' == mp.meta[1]['revision']
        assert b'5.34/23' == mp.meta[1]['root_version']
        assert mp.meta[1]['command'].startswith(b'JEnergy /pbs/throng/km3net')
        assert mp.meta[1]['command'].endswith(b'1 --!')

        assert b'JTriggerEfficiency' == mp.meta[-1]['application_name']
        assert b'8519' == mp.meta[-1]['revision']
        assert b'5.34/23' == mp.meta[-1]['root_version']
        assert mp.meta[-1]['command'].startswith(
            b'JTriggerEfficiency /pbs/throng/km3net'
        )
        assert mp.meta[-1]['command'].endswith(b'326 --!')

    def test_get_table(self):
        string = (
            b"A 123\nA 1.2.3\nA KM3NET\nA a\nA b\nA Linux\n"
            b"B 456\nB 4.5.6\nB KM3NET\nB c\nB Linux"
        )
        mp = MetaParser()
        mp.parse_string(string)
        tab = mp.get_table()
        assert 2 == len(tab)
        assert '/meta' == tab.h5loc
        assert tab.h5singleton
        assert b'B c' == tab['command'][1]
        assert np.dtype('S7') == tab['command'].dtype

    def test_empty_string(self):
        mp = MetaParser()
        mp.parse_string('')
        assert mp.get_table() is None
