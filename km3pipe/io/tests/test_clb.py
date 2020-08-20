# Filename: test_clb.py
# pylint: disable=C0111,R0904,C0103
"""
...

"""
from io import StringIO
from os.path import join, dirname

from km3pipe.testing import TestCase, data_path
from km3pipe.io.clb import CLBPump

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestCLBPump(TestCase):
    def setUp(self):
        self.pump = CLBPump(file=data_path("daq/clb.dqd"))

    def test_determine_packet_positions_finds_packets(self):
        self.assertListEqual([0, 1406, 2284], self.pump._packet_positions)

    def test_length(self):
        assert 3 == len(self.pump)

    def test_getindex(self):
        blob = self.pump[0]
        self.assertEqual(0, blob["PacketInfo"].run[0])
        self.assertEqual("TTDC", blob["PacketInfo"].data_type[0])
        assert 229 == len(blob["Hits"])
        a_pmt_data = blob["Hits"][2]
        self.assertEqual(15, a_pmt_data.channel_id)
        self.assertEqual(66541067, a_pmt_data.time)
        self.assertEqual(28, a_pmt_data.tot)

        assert 229 == len(self.pump[0]["Hits"])
        assert 141 == len(self.pump[1]["Hits"])
        assert 229 == len(self.pump[2]["Hits"])

    def test_iterator(self):
        i = 0
        for b in self.pump:
            i += 1
        assert 3 == i

        i = 0
        for b in self.pump:
            i += 1
        assert 3 == i
