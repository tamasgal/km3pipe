# Filename: pickle.py
# pylint: disable=locally-disabled
"""
Pump for the pickle data format.

"""
from __future__ import absolute_import, print_function, division

import pickle
from collections import namedtuple

from km3pipe.core import Pump
from km3pipe.logger import get_logger

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = get_logger(__name__)    # pylint: disable=C0103


class PicklePump(Pump):
    """A pump for Python pickles."""

    def configure(self):

        self.filename = self.get('filename')
        self.events = pickle.load(open(self.filename, "rb"))

    def get_blob(self, index):
        hits = self.events[0]
        t0 = min([hit[3] for hit in hits])
        raw_hits = []
        for hit in hits:
            raw_hit = RawHit(hit[0], hit[1], hit[2], hit[3] - t0)
            print(raw_hit)
            raw_hits.append(raw_hit)
        blob = {'EvtRawHits': raw_hits}
        return blob


RawHit = namedtuple('RawHit', 'id pmt_id tot time')
