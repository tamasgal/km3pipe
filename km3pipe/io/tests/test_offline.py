#!/usr/bin/env python3
from os.path import join, dirname

import km3pipe as kp
from km3pipe.testing import TestCase, data_path
from km3pipe.io.offline import OfflinePump


# class TestEventPump(TestCase):
#     def setUp(self):
#         self.pump = EventPump(
#             filename=data_path(
#                 "offline/mcv5.0.DAT004340.propa.sirene.jte.jchain.aanet.4340.root"
#             )
#         )

#     def test_iteration(self):
#         i = 0
#         for blob in self.pump:
#             i += 1

#         assert 3 == i

#     def test_getitem(self):
#         blob = self.pump[0]
#         assert "Header" in blob
#         assert "Hits" in blob
#         assert "McHits" in blob
#         assert "McTracks" in blob

#         assert 86 == len(blob["Hits"])

#         with self.assertRaises(IndexError):
#             self.pump[4]

#     def test_hits(self):
#         n_hits = [86, 111, 83]
#         i = 0
#         for blob in self.pump:
#             assert n_hits[i] == len(blob["Hits"])
#             i += 1

#     def test_mc_hits(self):
#         n_mc_hits = [147, 291, 160]
#         i = 0
#         for blob in self.pump:
#             assert n_mc_hits[i] == len(blob["McHits"])
#             i += 1

#     def test_mc_tracks(self):
#         n_mc_tracks = [4, 8, 36]
#         i = 0
#         for blob in self.pump:
#             assert n_mc_tracks[i] == len(blob["McTracks"])
#             i += 1

#     def test_header(self):
#         for blob in self.pump:
#             pass

#         self.assertListEqual([0, 1000000000.0, -1, -0.052], list(blob["Header"].cut_in))


class TestOfflinePump(TestCase):
    def setUp(self):
        self.pump = OfflinePump(
            filename=data_path(
                "offline/mcv5.0.DAT004340.propa.sirene.jte.jchain.aanet.4340.root"
            )
        )

    def test_offline_pump_iteration(self):
        i = 0
        for blob in self.pump:
            i += 1
        assert 3 == i

    def test_offline_pump_has_header(self):
        for blob in self.pump:
            assert "header" in blob

    def test_offline_pump_in_pipeline(self):
        class Observer(kp.Module):
            def configure(self):
                self.i = 0

            def process(self, blob):
                assert "header" in blob
                assert "event" in blob
                self.i += 1
                return blob

            def finish(self):
                return {"n_events": self.i}

        pipe = kp.Pipeline()
        pipe.attach(
            OfflinePump,
            filename=data_path(
                "offline/mcv5.0.DAT004340.propa.sirene.jte.jchain.aanet.4340.root"
            ),
        )
        pipe.attach(Observer)
        results = pipe.drain()

        assert 3 == results["Observer"]["n_events"]
