# Filename: common.py
# pylint: disable=locally-disabled
"""
A collection of commonly used modules.

"""
from time import time

import numpy as np

import km3pipe as kp
from km3pipe import Module, Blob
from km3pipe.tools import prettyln
from km3pipe.sys import peak_memory_usage
from km3pipe.math import zenith, azimuth
from km3pipe.dataclasses import Table

log = kp.logger.get(__name__)


class Wrap(Module):
    """Wrap a key-val dictionary as a Serialisable.
    """

    def configure(self):
        self.keys = self.get('keys') or None
        key = self.get('key') or None
        if key and not self.keys:
            self.keys = [key]

    def process(self, blob):
        keys = sorted(blob.keys()) if self.keys is None else self.keys
        for key in keys:
            dat = blob[key]
            if dat is None:
                continue
            dt = np.dtype([(f, float) for f in sorted(dat.keys())])
            arr = Table(dat, dtype=dt)
            blob[key] = arr
        return blob


class Dump(Module):
    """Print the content of the blob.

    Parameters
    ----------
    keys: collection(string), optional [default=None]
        Keys to print. If None, print all keys.
    full: bool, default=False
        Print blob values too, not just the keys?
    """

    def configure(self):
        self.keys = self.get('keys') or None
        self.full = self.get('full') or False
        key = self.get('key') or None
        if key and not self.keys:
            self.keys = [key]

    def process(self, blob):
        keys = sorted(blob.keys()) if self.keys is None else self.keys
        for key in keys:
            print(key + ':')
            if self.full:
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

    def configure(self):
        self.keys = self.get('keys') or set()
        key = self.get('key') or None
        if key and not self.keys:
            self.keys = [key]

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

    def configure(self):
        self.keys = self.get('keys') or set()
        key = self.get('key') or None
        if key and not self.keys:
            self.keys = [key]

    def process(self, blob):
        out = Blob()
        for key in blob.keys():
            if key in self.keys:
                out[key] = blob[key]
        return out


class HitCounter(Module):
    """Prints the number of hits"""

    def process(self, blob):
        try:
            print("Number of hits: {0}".format(len(blob['Hit'])))
        except KeyError:
            pass
        return blob


class HitCalibrator(Module):
    """A very basic hit calibrator, which requires a `Calibration` module."""
    def configure(self):
        self.input_key = self.get('input_key', default='Hits')
        self.output_key = self.get('output_key', default='CalibHits')

    def process(self, blob):
        if self.input_key not in blob:
            self.log.warn("No hits found in key '{}'.".format(self.input_key))
            return blob
        hits = blob[self.input_key]
        chits = self.calibration.apply(hits)
        blob[self.output_key] = chits
        return blob


class BlobIndexer(Module):
    """Puts an incremented index in each blob for the key 'blob_index'"""

    def configure(self):
        self.blob_index = 0

    def process(self, blob):
        blob['blob_index'] = self.blob_index
        self.blob_index += 1
        return blob


class StatusBar(Module):
    """Displays the current blob number."""

    def configure(self):
        self.iteration = 1

    def process(self, blob):
        prettyln("Blob {0:>7}".format(self.every * self.iteration))
        self.iteration += 1
        return blob

    def finish(self):
        prettyln(".", fill='=')


class TickTock(Module):
    """Display the elapsed time.

    Parameters
    ----------
    every: int, optional [default=1]
        Number of iterations between printout.
    """

    def configure(self):
        self.t0 = time()

    def process(self, blob):
        t1 = (time() - self.t0) / 60
        prettyln("Time/min: {0:.3f}".format(t1))
        return blob


class MemoryObserver(Module):
    """Shows the maximum memory usage

    Parameters
    ----------
    every: int, optional [default=1]
        Number of iterations between printout.
    """

    def process(self, blob):
        memory = peak_memory_usage()
        prettyln("Memory peak: {0:.3f} MB".format(memory))
        return blob


class Cut(Module):
    """Drop an event from the pipe, depending on some condition.

    Parameters
    ----------
    key: string
        Blob content to cut on (must have `.conv_to(to='pandas')` method)
    condition: string
        Condition to eval on the dataframe, e.g. "zenith >= 1.4".
    verbose: bool, optional (default=False)
        Print extra info?
    """

    def configure(self):
        self.key = self.require('key')
        self.cond = self.require('condition')
        self.verbose = self.get('verbose') or False

    def process(self, blob):
        df = blob[self.key].conv_to(to='pandas')
        ok = df.eval(self.cond).all()
        if not ok:
            if self.verbose:
                print('Condition "%s" not met, dropping...' % self.cond)
            return
        blob[self.key] = df
        return blob


class GetAngle(Module):
    """Convert pos(x, y, z) to zenith, azimuth."""

    def configure(self):
        self.key = self.require('key')

    def process(self, blob):
        df = blob[self.key].conv_to(to='pandas')
        df['zenith'] = zenith(df[['dir_x', 'dir_y', 'dir_z']])
        df['azimuth'] = azimuth(df[['dir_x', 'dir_y', 'dir_z']])
        blob[self.key] = df
        return blob


class Siphon(Module):
    """A siphon to accumulate a given volume of blobs.

    Parameters
    ----------
    volume: int
      number of blobs to hold
    flush: bool
      discard blobs after accumulation

    """

    def configure(self):
        self.volume = self.require('volume')  # [blobs]
        self.flush = self.get('flush', default=False)

        self.blob_count = 0

    def process(self, blob):
        self.blob_count += 1
        if self.blob_count > self.volume:
            log.debug("Siphone overflow reached!")
            if self.flush:
                log.debug("Flushing the siphon.")
                self.blob_count = 0
            return blob
