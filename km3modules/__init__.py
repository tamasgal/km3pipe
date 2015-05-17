# coding=utf-8
# Filename: __init__.py
# pylint: disable=locally-disabled
"""
A collection of commonly used modules.

"""
from __future__ import division, absolute_import, print_function

import timeit

from km3pipe import Module


class HitCounter(Module):
    """Prints the number of hits and raw hits in an Evt file"""
    def process(self, blob):
        try:
            print("Number of hits: {0}".format(len(blob['hit'])))
        except KeyError:
            pass
        try:
            print("Number of raw hits: {0}".format(len(blob['hit_raw'])))
        except KeyError:
            pass
        return blob


class StatusBar(Module):
    """Displays the current blob number"""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.blob_index = 0
        self.start = timeit.default_timer()

    def process(self, blob):
        print("------------[Blob {0:>7}]-------------".format(self.blob_index))
        self.blob_index += 1
        return blob

    def finish(self):
        """Display some basic statistics like elapsed time"""
        elapsed_time = timeit.default_timer() - self.start
        print("\n" + '='*42)
        print("Processed {0} blobs in {1} s."
              .format(self.blob_index, elapsed_time))


