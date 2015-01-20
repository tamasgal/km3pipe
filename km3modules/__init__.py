# coding=utf-8
# Filename: __init__.py
# pylint: disable=locally-disabled
"""
A collection of commonly used modules.

"""
from __future__ import division, absolute_import, print_function

import datetime

from km3pipe import Module
from km3pipe.hardware import Detector


class HitCounter(Module):
    """Prints the number of hits and raw hits in an Evt file"""
    def process(self, blob):
        print("Number of hits: {0}".format(len(blob['hit'])))
        print("Number of raw hits: {0}".format(len(blob['hit_raw'])))
        return blob


class StatusBar(Module):
    """Displays the current blob number"""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.blob_index = 0


        self.start = datetime.datetime.now()

    def process(self, blob):
        print("----[Blob {0:>5}]----".format(self.blob_index))
        self.blob_index += 1
        return blob

    def finish(self):
        """Display some basic statistics like elapsed time"""
        elapsed_time = self.start - datetime.datetime.now()
        print("\n" + '='*42)
        print("Processed {0} blobs in {1} s."
              .format(self.blob_index, elapsed_time.microseconds / 1e6))


class Geometry(Module):
    """A very simple, preliminary Module which gives access to the geometry"""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        filename = self.get('filename')
        self.detector = Detector(filename)

    def process(self, blob):
        blob['Detector'] = self.detector
        return blob