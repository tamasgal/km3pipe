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

        self.filename = self.get('filename')
        self.filenames = self.get('filenames') or []
        self.indices = self.get('indices')

        if not self.filename and not self.filenames:
            raise ValueError("No filename(s) defined")

        if self.filename:
            if "[index]" in self.filename and self.indices:
                self._parse_filenames()
            else:
                self.filenames.append(self.filename)

        self.blobs = self.blob_generator()

    def _parse_filenames(self):
        prefix, suffix = self.filename.split('[index]')
        self.filenames += [prefix + str(i) + suffix for i in self.indices]

    def get_blob(self, index):
        NotImpelementedYet("Aanet currently does not support indexing.")

    def blob_generator(self):
        """Create a blob generator."""
        # pylint: disable:F0401,W0612
        import aa
        from ROOT import EventFile 

        for filename in self.filenames:
            print("Reading from file: {0}".format(filename))
            event_file = EventFile(filename)
            for event in event_file:
                blob = {'Evt': event,
                        'RawHits': event.hits,
                        'MCHits': event.mc_hits,
                        'RecoTracks': event.trks,
                        'MCTracks': event.mc_trks}
                yield blob
            del event_file

    def process(self, blob):
        return next(self.blobs)

