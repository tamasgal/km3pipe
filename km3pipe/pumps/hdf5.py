#!/usr/bin/env python2
# Filename: numpy.py
# pylint: disable=locally-disabled
# vim:set ts=4 sts=4 sw=4 et:
"""Pump for Numpy arrays.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import tables

from km3pipe.core import Pump, Blob
from km3pipe.logger import logging


log = logging.getLogger(__name__)


class HDF5Pump(Pump):
    """HDF5Pump streams events from HDF5 tables.
    """

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get("filename")
        self.location = self.get("location")
        self.title = self.get("title")

        if not self.title:
            self.title = "data"
        if self.filename:
            self.h5file = tables.open_file(self.filename, 'r',
                                           title=self.title)
        else:
            log.warn("No filename specified. Take care of the file handling!")

    def finish(self):
        self.h5file.close()
