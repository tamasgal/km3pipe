# coding=utf-8
# Filename: aanet.py
# pylint: disable=locally-disabled
"""
Pump for the Aanet data format.

"""
from __future__ import division, absolute_import, print_function

from km3pipe import Pump
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103


class AanetPump(Pump):
    """A pump for binary Aanet files."""

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)

        # pylint: disable:F0401,W0612
        import aa
        from ROOT import EventFile

        self.filename = self.get('filename')
        if not self.filename:
            raise ValueError("No filename defined")
        self.event_file = EventFile(self.filename)
        self.blobs = self.blob_generator()

    def get_blob(self, index):
        NotImpelementedYet("Aanet currently does not support indexing.")

    def blob_generator(self):
        """Create a blob generator."""
        for event in self.event_file:
            blob = {'Evt': event,
                    'RawHits': event.hits,
                    'MCHits': event.mc_hits,
                    'RecoTracks': event.trks,
                    'MCTracks': event.mc_trks}
            yield blob

    def process(self, blob):
        return next(self.blobs)
