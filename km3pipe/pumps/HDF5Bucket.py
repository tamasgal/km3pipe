#!/bin/env python2
# vim:set ts=4 sts=4 sw=4 et:

from collections import defaultdict

import numpy as np
import tables

from km3pipe.core import Module


class HDF5Bucket(Module):
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get("filename")
        self.prefix = self.get("prefix") or '/'
        self.store = defaultdict(list)

    def process(self, blob):
        for key, val in blob.items():
            self.store[key].append(val)

    def finish(self):
        h5 = tables.open_file(self.filename, mode='w', ftitle="Data")
        for key, data in self.store.items():
            arr = np.array(data, dtype=[(key, float), ])
            h5.create_table(self.prefix, key, arr)
        h5.close()
