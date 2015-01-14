# coding=utf-8
# Filename: evt.py
# pylint: disable=locally-disabled
"""
Pumps for the EVT simulation dataformat.

"""
from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

from km3pipe import Pump
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103


class EvtPump(Pump):
    """Provides a pump for EVT-files"""

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename')

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
        self._rebuild_offsets()

    def extract_header(self):
        """Create a dictionary with the EVT header information"""
        raw_header = {}
        for line in self.blob_file:
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
        self.blob_file.seek(self.event_offsets[index], 0)
        blob = self._create_blob()
        return blob

    def process(self, blob):
        """Pump the next blob to the modules"""
        blob = self.get_blob(self.index)
        self.index += 1
        return blob

    def _rebuild_offsets(self):
        self.blob_file.seek(0, 0)
        self.event_offsets = []
        for line in self.blob_file:
            line = line.strip()
            if line.startswith('end_event:'):
                self._record_offset()
        self.event_offsets.pop()  # get rid of the last one
        self.blob_file.seek(self.event_offsets[0], 0)

    def _record_offset(self):
        """Stores the current file pointer position"""
        offset = self.blob_file.tell()
        self.event_offsets.append(offset)

    def _create_blob(self):
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
                if tag in ('neutrino', 'track_in', 'hit'):
                    values = [float(x) for x in value.split()]
                    blob.setdefault(tag, []).append(values)
                else:
                    blob[tag] = value.split()

    def finish(self):
        """Clean everything up"""
        self.blob_file.close()
