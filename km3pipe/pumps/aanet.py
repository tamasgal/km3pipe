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
        from ROOT import TFile, Evt, Trk
        
        self.filename = self.get('filename')
        self.treename = self.get('treename') or "E"
        if not self.filename:
            raise ValueError("No filename defined")
        self.index = 0
        self.rootfile = TFile(self.filename)
        self.evt = Evt()
        self.E = self.rootfile.Get(self.treename)
	self.N = self.E.GetEntries()
        self.E.SetBranchAddress('Evt', self.evt)

    def get_blob(self, index):
        self.E.GetEntry(index)
        return {'Evt': self.evt,
                'hits': self.evt.hits}

    def process(self, blob):
        self.E.GetEntry(self.index)
        if self.index == self.N:
            raise StopIteration
        else:
	    self.index += 1
            return {'Evt': self.evt}
