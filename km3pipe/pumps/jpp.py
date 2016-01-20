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

        self.index = self.get('index') or 0

        import aa  # noqa
        import ROOT
        if self.get('index'):
            self.index = self.get('index')
        else:
            self.index = 0

        self.index_start = self.get('index_start') or 1
        self.index_stop = self.get('index_stop') or 1

        self.filename = self.get('filename')
        self.basename = self.get('basename')
        if not self.filename and not self.basename:
            raise ValueError("No filename defined")

        self.file_index = self.index_start

        if self.basename:
            filename = self.basename + str(self.file_index) + ".JTE.root"
            self.rootfile = ROOT.EventFile(filename)

        else:
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
            self.file_index += 1
            if self.basename and self.file_index <= self.index_stop:
                import aa  # noqa
                import ROOT
                print("open next file")
                filename = self.basename + str(self.file_index) + ".JTE.root"
                self.rootfile = ROOT.EventFile(filename)
                self.index = 0
                self.process(blob)
            else:
                raise StopIteration

    def finish(self):
        self.rootfile.Close()
