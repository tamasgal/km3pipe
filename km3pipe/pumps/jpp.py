#!/usr/bin/env python
# coding=utf-8
# Filename: jpp.py
# pylint: disable=
"""
Pump for the jpp file read through aanet interface.

"""
from __future__ import division, absolute_import, print_function

from km3pipe import Pump
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103


class JPPPump(Pump):
    """A pump for JPP ROOT files."""

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)

        import aa
        import ROOT

        self.filename = self.get('filename')
        if not self.filename:
            raise ValueError("No filename defined")
            
        if self.get('index'):
            self.index = self.get('index')
        else:
            self.index = 0
            
        self.rootfile = ROOT.EventFile(self.filename)
        self.evt = ROOT.Evt()

    def get_blob(self, index):
        """Return the blob"""
        self.rootfile.set_index(index)
        self.evt = self.rootfile.evt
        return {'Evt': self.evt}

    def process(self, blob):
        if self.rootfile.set_index(self.index):
            self.evt = self.rootfile.evt
            self.index += 1
            return {'Evt': self.evt}
        else:
            raise StopIteration


    def finish(self):
        self.rootfile.Close()
