# coding=utf-8
# Filename: evt.py
# pylint: disable=locally-disabled
"""
Pumps for the EVT simulation dataformat.

"""
from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

import sys

from km3pipe import Pump
from km3pipe.dataclasses import Hit, RawHit, Track, Neutrino
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103


class EvtPump(Pump):
    """Provides a pump for EVT-files"""

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename')
        self.cache_enabled = self.get('cache_enabled') or False

        self.raw_header = None
        self.event_offsets = []
        self.index = 0

        if self.filename:
            self.open_file(self.filename)
            self.prepare_blobs()
        else:
            log.warn("No filename specified. Take care of the file handling!")

    def prepare_blobs(self):
        """Populate the blobs"""
        self.raw_header = self.extract_header()
        if self.cache_enabled:
            self._cache_offsets()

    def extract_header(self):
        """Create a dictionary with the EVT header information"""
        raw_header = {}
        #for line in self.blob_file:
        for line in iter(self.blob_file.readline, ''):
            line = line.strip()
            try:
                tag, value = line.split(':')
            except ValueError:
                continue
            raw_header[tag] = value.split()
            if line.startswith('end_event:'):
                self._record_offset()
                return raw_header
        raise ValueError("Incomplete header, no 'end_event' tag found!")

    def get_blob(self, index):
        """Return a blob with the event at the given index"""
        if index > len(self.event_offsets) - 1:
            self._cache_offsets(index, verbose=False)
        self.blob_file.seek(self.event_offsets[index], 0)
        blob = self._create_blob()
        if blob is None:
            raise IndexError
        else:
            return blob

    def process(self, blob=None):
        """Pump the next blob to the modules"""
        try:
            blob = self.get_blob(self.index)
        except IndexError:
            raise StopIteration
        self.index += 1
        return blob

    def _cache_offsets(self, up_to_index=None, verbose=True):
        if not up_to_index:
            if verbose:
                print("Caching event file offsets, this may take a minute.")
            self.blob_file.seek(0, 0)
            self.event_offsets = []
        else:
            self.blob_file.seek(self.event_offsets[-1], 0)
        for line in iter(self.blob_file.readline, ''):
            line = line.strip()
            if line.startswith('end_event:'):
                self._record_offset()
                if len(self.event_offsets) % 100 == 0:
                    if verbose:
                        print('.', end='')
                    sys.stdout.flush()
            if up_to_index and len(self.event_offsets) >= up_to_index + 1:
                return
        self.event_offsets.pop()  # get rid of the last entry
        #self.blob_file.seek(self.event_offsets[0], 0)
        print("\n{0} events indexed.".format(len(self.event_offsets)))

    def _record_offset(self):
        """Stores the current file pointer position"""
        offset = self.blob_file.tell()
        self.event_offsets.append(offset)

    def _create_blob(self):
        """Parse the next event from the current file position"""
        blob = None
        for line in self.blob_file:
            line = line.strip()
            if line.startswith('end_event:') and blob:
                blob['raw_header'] = self.raw_header
                return blob
            if line.startswith('start_event:'):
                blob = {}
                tag, value = line.split(':')
                blob[tag] = value.split()
                continue
            if blob:
                tag, value = line.split(':')
                if tag in ('track_in', 'hit', 'hit_raw'):
                    values = [float(x) for x in value.split()]
                    blob.setdefault(tag, []).append(values)
                    if tag == 'hit':
                        blob.setdefault("EvtHits", []).append(Hit(*values))
                    if tag == "hit_raw":
                        blob.setdefault("EvtRawHits", []).append(RawHit(*values))
                    if tag == "track_in":
                        blob.setdefault("TrackIns", []).append(Track(*values))
                else:
                    if tag == 'neutrino':
                        values = [float(x) for x in value.split()]
                        blob['Neutrino'] = Neutrino(*values)
                    else:
                        blob[tag] = value.split()

    def finish(self):
        """Clean everything up"""
        self.blob_file.close()
