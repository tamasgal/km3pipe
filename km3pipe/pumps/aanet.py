# coding=utf-8
# Filename: aanet.py
# pylint: disable=locally-disabled
"""
Pump for the Aanet data format.

"""
from __future__ import division, absolute_import, print_function

from km3pipe import Pump
from km3pipe.logger import logging
import os.path

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

        self.header = None
        self.blobs = self.blob_generator()

    def _parse_filenames(self):
        prefix, suffix = self.filename.split('[index]')
        self.filenames += [prefix + str(i) + suffix for i in self.indices]

    def get_blob(self, index):
        NotImplementedError("Aanet currently does not support indexing.")

    def blob_generator(self):
        """Create a blob generator."""
        # pylint: disable:F0401,W0612
        import aa  # noqa
        from ROOT import EventFile

        for filename in self.filenames:
            print("Reading from file: {0}".format(filename))
            if not os.path.exists(filename):
                log.warn(filename + " not available: continue without it")
                continue

            try:
                event_file = EventFile(filename)
            except Exception:
                raise SystemExit("Could not open file")

            try:
                self.header = event_file.rootfile().Get("Header")
            except AttributeError:
                pass

            for event in event_file:
                blob = {'Evt': event,
                        'Hits': event.hits,
                        'MCHits': event.mc_hits,
                        'Tracks': event.trks,
                        'MCTracks': event.mc_trks,
                        'filename': filename,
                        'Header': self.header}
                yield blob
            del event_file

    def process(self, blob):
        return next(self.blobs)

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        return next(self.blobs)
