# coding=utf-8
# Filename: common.py
# pylint: disable=locally-disabled
"""
A collection of commonly used modules.

"""
from __future__ import division, absolute_import, print_function

import numpy as np

from km3pipe import Module, SkipEvent
from km3pipe.tools import peak_memory_usage, zenith, azimuth
from km3pipe.dataclasses import KM3DataFrame, KM3Array     # noqa


class Wrap(Module):
    """Wrap a key-val dictionary as a Serialisable.
    """
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.keys = self.get('keys') or None

    def process(self, blob):
        keys = sorted(blob.keys()) if self.keys is None else self.keys
        for key in keys:
            dat = blob[key]
            if dat is None:
                continue
            dt = np.dtype([(f, float) for f in sorted(dat.keys())])
            arr = KM3Array.from_dict(dat, dt)
            blob[key] = arr
        return blob


class Dump(Module):
    """Print the content of the blob.

    If no argument is specified, dump everything.

    Parameters
    ----------
    keys: collection(string), optional
        Keys to print.
    all: bool, default=False
        Print blob values too, not just the keys?
    """
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.keys = self.get('keys') or None
        self.all = self.get('all') or False

    def process(self, blob):
        keys = sorted(blob.keys()) if self.keys is None else self.keys
        for key in keys:
            print(key, end=': ')
            if self.all:
                print(blob[key].__repr__())
                print('')
        print('----------------------------------------\n')
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
        self.keys = self.get('keys') or set()

    def process(self, blob):
        for key in self.keys:
            blob.pop(key, None)
        return blob


class Keep(Module):
    """Keep only specified keys in the blob.

    Parameters
    ----------
    keys: collection(string), optional
        Keys to keep. Everything else is removed.
    """
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.keys = self.get('keys') or set()

    def process(self, blob):
        for key in blob.keys():
            if key not in self.keys:
                blob.pop(key, None)
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
            print('-'*23 + "[Blob {0:>7}]".format(self.blob_index) + '-'*23)
        self.blob_index += 1
        return blob

    def finish(self):
        print('=' * 60)


class MemoryObserver(Module):
    """Shows the maximum memory usage"""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)

    def process(self, blob):
        memory = peak_memory_usage()
        print("Memory peak usage: {0:.3f} MB".format(memory))


class Cut(Module):
    """Drop an event from the pipe if certain criteria are met.

    This is handled by the ``km3pipe.core.SkipException``.
    """
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.key = self.get('key')
        self.cond = self.get('condition')

    def process(self, blob):
        df = blob[self.key]
        ok = df.eval(self.cond).all()
        if not ok:
            raise SkipEvent
        df[self.key] = df
        return blob


class GetAngle(Module):
    """Convert pos(x, y, z) to zenith, azimuth."""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.key = self.get('key')

    def process(self, blob):
        df = blob[self.key].conv_to(to='pandas')
        df['zenith'] = zenith(df['dir_x'], df['dir_y'], df['dir_z'])
        df['azimuth'] = azimuth(df['dir_x'], df['dir_y'], df['dir_z'])
        return blob
