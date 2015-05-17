# coding=utf-8
# Filename: __init__.py
# pylint: disable=locally-disabled
"""
A collection of commonly used modules.

"""
from __future__ import division, absolute_import, print_function

import timeit

from km3pipe import Module


class CHGrabber(Module):
    """Grabs controlhost message for a given tag."""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)

        self.host = self.get('host') or '127.0.0.1'
        self.port = self.get('port') or 5553
        self.tag = self.get('tag') or 'MSG'
        self.key_for_data = self.get('key_for_data') or 'CHData'
        self.key_for_prefix = self.get('key_for_prefix') or 'CHPrefix'

        self.ch_client = None

        self._init_controlhost()

    def _init_controlhost(self):
        """Set up the controlhost connection"""
        from controlhost import Client
        self.ch_client = Client(self.host, self.port)
        self.ch_client._connect()
        self.ch_client.subscribe(self.tag)

    def process(self, blob):
        """Wait for the next packet and put it in the blob"""
        prefix, data = self.ch_client.get_message()
        blob[self.key_for_prefix] = prefix
        blob[self.key_for_data] = data
        return blob
        
    def finish(self):
        """Clean up the JLigier controlhost connection"""
        self.ch_client._disconnect()


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


