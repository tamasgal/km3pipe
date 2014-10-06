# coding=utf-8
# Filename: evt.py
# pylint: disable=locally-disabled
"""
Pumps for the EVT simulation dataformat.

"""
from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

from km3pipe import Pump
from km3pipe.logger import get_logger

log = get_logger(__name__)  # pylint: disable=C0103


class EvtPump(Pump):
    """Provides a pump for EVT-files"""

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename')

        self.blobs = None
        self.raw_header = None

        if self.filename:
            self.open_file(self.filename)
            self.prepare_blobs()
        else:
            log.warn("No filename specified. Take care of the file handling!")

    def prepare_blobs(self):
        """Populate the blobs"""
        self.raw_header = self.extract_header()
        self.blobs = self.blob_generator()

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
                return raw_header
        raise ValueError("Incomplete header, no 'end_event' tag found!")

    def blob_generator(self):
        """Create a generator object which extracts events from an EVT file."""
        blob = None
        for line in self.blob_file:
            line = line.strip()
            if line.startswith('end_event:') and blob:
                yield blob
                blob = None
                continue
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

    def process(self, blob):
        """Pump the next blob to the modules"""
        return next(self.blobs)

    def finish(self):
        """Clean everything up"""
        self.blob_file.close()
