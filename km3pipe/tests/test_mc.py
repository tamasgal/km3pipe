# Filename: test_mc.py
# pylint: disable=locally-disabled,C0111,R0904,C0103

from km3pipe.testing import TestCase
from km3pipe.mc import geant2pdg, pdg2name

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
        self.assertEqual('mu-', pdg2name(13))
        self.assertEqual('anu_tau', pdg2name(-16))

    def test_pdg2name_returns_NA_for_unknown_particle(self):
        self.assertEqual('N/A', pdg2name(0))
