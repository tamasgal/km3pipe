# coding=utf-8
# Filename: __init__.py
# pylint: disable=locally-disabled
"""
A collection of commonly used modules.

"""
from __future__ import division, absolute_import, print_function
from pprint import pprint

from km3pipe import Module
from km3pipe.tools import peak_memory_usage


class Dump(Module):
    """Print the content of the blob.

    If no argument is specified, dump everything.

    Parameters
    ----------
    keys: collection(string), optional
        Keys to print.
    """
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.keys = self.get('keys') or None

    def process(self, blob):
        pprint(blob)
        return blob


class Delete(Module):
    """Remove specific keys from the blob.

    Parameters
    ----------
    keys: collection(string), optional
        Keys to remove.
    """
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.todelete = self.get('keys') or set()

    def process(self, blob):
        for key in self.keys:
            try:
                del blob[key]
            except KeyError:
                pass
        return blob


class HitCounter(Module):
    """Prints the number of hits"""
    def process(self, blob):
        try:
            print("Number of hits: {0}".format(len(blob['Hit'])))
        except KeyError:
            pass
        return blob


class BlobIndexer(Module):
    """Puts an incremented index in each blob for the key 'blob_index'"""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.blob_index = 0

    def process(self, blob):
        blob['blob_index'] = self.blob_index
        self.blob_index += 1
        return blob


class StatusBar(Module):
    """Displays the current blob number"""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.every = self.get('every') or 1
        self.blob_index = 0

    def process(self, blob):
        if self.blob_index % self.every == 0:
            print('-'*33 + "[Blob {0:>7}]".format(self.blob_index) + '-'*33)
        self.blob_index += 1
        return blob


class MemoryObserver(Module):
    """Shows the maximum memory usage"""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)

    def process(self, blob):
        memory = peak_memory_usage()
        print("Memory peak usage: {0:.3f} MB".format(memory))
