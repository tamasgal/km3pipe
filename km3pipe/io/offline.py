#!/usr/bin/env python3

from thepipe import Module, Blob
from ..dataclasses import Table
from .hdf5 import HDF5Header
from thepipe import Provenance

import km3io
import numpy as np
from collections import defaultdict


class OfflinePump(Module):
    def configure(self):
        self._filename = self.get("filename")

        self._reader = km3io.OfflineReader(self._filename)
        self.header = self._reader.header
        self.blobs = self._blob_generator()

        Provenance().record_input(
            self._filename, uuid=self._reader.uuid, comment="OfflinePump input"
        )

        self.expose(self.header, "offline_header")

    def process(self, blob=None):
        return next(self.blobs)

    def finish(self):
        self._reader.close()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.blobs)

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError("Only integer indices are supported.")
        return Blob({"event": self._reader.events[item], "header": self.header})

    def get_number_of_blobs(self):
        return len(self._reader.events)

    def _blob_generator(self):
        for event in self._reader.events:
            blob = Blob({"event": event, "header": self.header})
            yield blob
