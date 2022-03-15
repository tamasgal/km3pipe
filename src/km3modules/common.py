# Filename: common.py
# -*- coding: utf-8 -*-
# pylint: disable=locally-disabled
"""
A collection of commonly used modules.

"""

import sqlite3
from time import time

import numpy as np

import km3pipe as kp
from km3pipe import Module, Blob
from km3pipe.tools import prettyln
from km3pipe.sys import peak_memory_usage

log = kp.logger.get_logger(__name__)


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
        self.keys = self.get("keys") or None
        self.full = self.get("full") or False
        key = self.get("key") or None
        if key and not self.keys:
            self.keys = [key]

    def process(self, blob):
        keys = sorted(blob.keys()) if self.keys is None else self.keys
        for key in keys:
            print(key + ":")
            if self.full:
                print(blob[key].__repr__())
            print("")
        print("----------------------------------------\n")
        return blob


class Delete(Module):
    """Remove specific keys from the blob.

    Parameters
    ----------
    keys: collection(string), optional
        Keys to remove.
    """

    def configure(self):
        self.keys = self.get("keys") or set()
        key = self.get("key") or None
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
        self.keys = self.get("keys", default=set())
        key = self.get("key", default=None)
        self.h5locs = self.get("h5locs", default=set())
        if key and not self.keys:
            self.keys = [key]

    def process(self, blob):
        out = Blob()
        for key in blob.keys():
            if key in self.keys:
                out[key] = blob[key]
            elif hasattr(blob[key], "h5loc") and blob[key].h5loc.startswith(
                tuple(self.h5locs)
            ):
                out[key] = blob[key]
        return out


class HitCounter(Module):
    """Prints the number of hits"""

    def process(self, blob):
        try:
            self.cprint("Number of hits: {0}".format(len(blob["Hit"])))
        except KeyError:
            pass
        return blob


class HitCalibrator(Module):
    """A very basic hit calibrator, which requires a `Calibration` module."""

    def configure(self):
        self.input_key = self.get("input_key", default="Hits")
        self.output_key = self.get("output_key", default="CalibHits")

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
        blob["blob_index"] = self.blob_index
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
        prettyln(".", fill="=")


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
        self.volume = self.require("volume")  # [blobs]
        self.flush = self.get("flush", default=False)

        self.blob_count = 0

    def process(self, blob):
        self.blob_count += 1
        if self.blob_count > self.volume:
            log.debug("Siphone overflow reached!")
            if self.flush:
                log.debug("Flushing the siphon.")
                self.blob_count = 0
            return blob


class MultiFilePump(kp.Module):
    """Use the given pump to iterate through a list of files.

    The group_id will be reset so that it's unique for each iteration.

    Parameters
    ----------
    pump: Pump
      The pump to be used to generate the blobs.
    filenames: iterable(str)
      List of filenames.
    kwargs: dict(str -> any) optional
      Keyword arguments to be passed to the pump.

    """

    def configure(self):
        self.pump = self.require("pump")
        self.filenames = self.require("filenames")
        self.kwargs = self.get("kwargs", default={})
        self.blobs = self.blob_generator()
        self.cprint("Iterating through {} files.".format(len(self.filenames)))
        self.n_processed = 0
        self.group_id = 0

    def blob_generator(self):
        for filename in self.filenames:
            self.cprint("Current file: {}".format(filename))
            pump = self.pump(filename=filename, **self.kwargs)
            for blob in pump:
                self._set_group_id(blob)
                blob["filename"] = filename
                yield blob
                self.group_id += 1
            self.n_processed += 1

    def _set_group_id(self, blob):
        for key, entry in blob.items():
            if isinstance(entry, kp.Table):
                if hasattr(entry, "group_id"):
                    entry.group_id = self.group_id
                else:
                    blob[key] = entry.append_columns("group_id", self.group_id)

    def process(self, blob):
        return next(self.blobs)

    def finish(self):
        self.cprint(
            "Fully processed {} out of {} files.".format(
                self.n_processed, len(self.filenames)
            )
        )


class LocalDBService(kp.Module):
    """Provides a local sqlite3 based database service to store information"""

    def configure(self):
        self.filename = self.require("filename")
        self.thread_safety = self.get("thread_safety", default=True)
        self.connection = None

        self.expose(self.create_table, "create_table")
        self.expose(self.table_exists, "table_exists")
        self.expose(self.insert_row, "insert_row")
        self.expose(self.query, "query")

        self._create_connection()

    def _create_connection(self):
        """Create database connection"""
        try:
            self.connection = sqlite3.connect(
                self.filename, check_same_thread=self.thread_safety
            )
            self.cprint(sqlite3.version)
        except sqlite3.Error as exception:
            self.log.error(exception)

    def query(self, query):
        """Execute a SQL query and return the result of fetchall()"""
        cursor = self.connection.cursor()
        cursor.execute(query)
        return cursor.fetchall()

    def insert_row(self, table, column_names, values):
        """Insert a row into the table with a given list of values"""
        cursor = self.connection.cursor()
        query = "INSERT INTO {} ({}) VALUES ({})".format(
            table, ", ".join(column_names), ",".join("'" + str(v) + "'" for v in values)
        )
        cursor.execute(query)
        self.connection.commit()

    def create_table(self, name, columns, types, overwrite=False):
        """Create a table with given columns and types, overwrite if specified


        The `types` should be a list of SQL types, like ["INT", "TEXT", "INT"]
        """
        cursor = self.connection.cursor()
        if overwrite:
            cursor.execute("DROP TABLE IF EXISTS {}".format(name))

        cursor.execute(
            "CREATE TABLE {} ({})".format(
                name, ", ".join(["{} {}".format(*c) for c in zip(columns, types)])
            )
        )
        self.connection.commit()

    def table_exists(self, name):
        """Check if a table exists in the database"""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT count(name) FROM sqlite_master "
            "WHERE type='table' AND name='{}'".format(name)
        )
        return cursor.fetchone()[0] == 1

    def finish(self):
        if self.connection:
            self.connection.close()


class Observer(kp.Module):
    """A simple helper to observe the blobs in a test pipeline.

    Parameters
    ----------
    count: int
      The exact number of iterations the pipeline has to drain
    required_keys: list(str)
      A list of keys which has to be present in a blob in every cycle.
    """

    def configure(self):
        self.count = self.get("count")
        self.required_keys = self.get("required_keys", default=[])
        self._count = 0

    def process(self, blob):
        self._count += 1
        for key in self.required_keys:
            assert key in blob
        return blob

    def finish(self):
        print(f"Target count={self._count}, actual count={self.count}")
        if self.count is not None:
            assert self.count == self._count


class FilePump(kp.Module):
    """A basic iterator for a list of files.

    Parameters
    ----------
    filenames: iterable(str)
      The filenames to be iterated over which are put into ``blob["filename"]``

    """

    def configure(self):
        self.filenames = self.require("filenames")
        self.blobs = self.blob_generator()

    def blob_generator(self):
        for filename in self.filenames:
            yield kp.Blob({"filename": filename})

    def process(self, blob):
        blob.update(next(self.blobs))
        return blob
