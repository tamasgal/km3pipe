# Filename: test_mc.py
# pylint: disable=locally-disabled,C0111,R0904,C0103
import numpy as np

from km3pipe import Table, Blob
from km3pipe.testing import TestCase
from km3pipe.mc import geant2pdg, pdg2name, convert_mc_times_to_jte_times

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestMc(TestCase):
    def test_geant2pdg(self):
        self.assertEqual(22, geant2pdg(1))
        self.assertEqual(-13, geant2pdg(5))

    def test_geant2pdg_returns_0_for_unknown_particle_id(self):
        self.assertEqual(0, geant2pdg(-999))

    def test_pdg2name(self):
        self.assertEqual("mu-", pdg2name(13))
        self.assertEqual("nu(tau)~", pdg2name(-16))

    def test_pdg2name_returns_NA_for_unknown_particle(self):
        self.assertEqual("N/A", pdg2name(0))


class TestMCConvert(TestCase):
    def setUp(self):
        self.event_info = Table(
            {
                "timestamp": 1,
                "nanoseconds": 700000000,
                "mc_time": 1.74999978e9,
            }
        )

        self.mc_tracks = Table(
            {
                "time": 1,
            }
        )

        self.mc_hits = Table(
            {
                "time": 30.79,
            }
        )

        self.blob = Blob(
            {
                "event_info": self.event_info,
                "mc_hits": self.mc_hits,
                "mc_tracks": self.mc_tracks,
            }
        )

    def test_convert_mc_times_to_jte_times(self):
        times_mc_tracks = convert_mc_times_to_jte_times(
            self.mc_tracks.time,
            self.event_info.timestamp * 1e9 + self.event_info.nanoseconds,
            self.event_info.mc_time,
        )
        times_mc_hits = convert_mc_times_to_jte_times(
            self.mc_hits.time,
            self.event_info.timestamp * 1e9 + self.event_info.nanoseconds,
            self.event_info.mc_time,
        )

        assert times_mc_tracks is not None
        assert times_mc_hits is not None
        print(times_mc_tracks, times_mc_hits)
        assert np.allclose(times_mc_tracks, 49999781)
        assert np.allclose(times_mc_hits, 49999810.79)
