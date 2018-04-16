# Filename: test_time.py
# pylint: disable=locally-disabled,C0111,R0904,C0103

import km3pipe as kp
from km3modules.common import Siphon
from km3pipe.testing import TestCase, MagicMock

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestSiphon(TestCase):
    def test_siphon(self):
        class APump(kp.Pump):
            def configure(self):
                self.i = 0

            def process(self, blob):
                self.i += 1
                blob['i'] = self.i
                return blob

        class Observer(kp.Module):
            def configure(self):
                self.mock = MagicMock()

            def process(self, blob):
                self.mock()
                return blob

            def finish(self):
                assert self.mock.call_count == 7

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Siphon, volume=10)
        pipe.attach(Observer)
        pipe.drain(17)

    def test_siphon_with_flush(self):
        class APump(kp.Pump):
            def configure(self):
                self.i = 0

            def process(self, blob):
                self.i += 1
                blob['i'] = self.i
                return blob

        class Observer(kp.Module):
            def configure(self):
                self.mock = MagicMock()

            def process(self, blob):
                self.mock()
                return blob

            def finish(self):
                assert self.mock.call_count == 1

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Siphon, volume=10, flush=True)
        pipe.attach(Observer)
        pipe.drain(21)

    def test_siphon_with_flush_2(self):
        class APump(kp.Pump):
            def configure(self):
                self.i = 0

            def process(self, blob):
                self.i += 1
                blob['i'] = self.i
                return blob

        class Observer(kp.Module):
            def configure(self):
                self.mock = MagicMock()

            def process(self, blob):
                self.mock()
                return blob

            def finish(self):
                assert self.mock.call_count == 2

        pipe = kp.Pipeline()
        pipe.attach(APump)
        pipe.attach(Siphon, volume=10, flush=True)
        pipe.attach(Observer)
        pipe.drain(22)
