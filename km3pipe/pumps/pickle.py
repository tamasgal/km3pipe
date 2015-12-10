# coding=utf-8
# Filename: pickle.py
# pylint: disable=locally-disabled
"""
Pump for the pickle data format.

"""
from __future__ import division, absolute_import, print_function

import pickle
from collections import namedtuple

from km3pipe import Pump
from km3pipe.logger import logging


log = logging.getLogger(__name__)  # pylint: disable=C0103


class PicklePump(Pump):
    """A pump for Python pickles."""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)

        self.filename = self.get('filename')
        self.events = pickle.load(open(self.filename, "rb" ))

    def get_blob(self, index):
        hits = self.events[0]
        raw_hits
        for hit in hits:
            pass
        blob = {}
        return blob


RawHit = namedtuple('RawHit', 'id pmt_id tot time')
