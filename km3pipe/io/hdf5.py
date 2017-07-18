# coding=utf-8
# Filename: hdf5.py
# pylint: disable=C0103,R0903
# vim:set ts=4 sts=4 sw=4 et:
"""
Read and write KM3NeT-formatted HDF5 files.
"""
from __future__ import division, absolute_import, print_function

from collections import OrderedDict, defaultdict
import os.path
from six import itervalues, iteritems
import warnings

import numpy as np
import tables as tb

import km3pipe as kp
from km3pipe.core import Pump, Module, Blob
from km3pipe.dataclasses import (KM3Array, KM3DataFrame,
                                 RawHitSeries, CRawHitSeries,
                                 McHitSeries, CMcHitSeries, deserialise_map)
from km3pipe.logger import logging
from km3pipe.dev import camelise, decamelise, split

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"

FORMAT_VERSION = np.string_('4.1')
MINIMUM_FORMAT_VERSION = np.string_('4.1')


class H5VersionError(Exception):
    pass


def check_version(h5file, filename):
    try:
        version = np.string_(h5file.root._v_attrs.format_version)
    except AttributeError:
        log.error("Could not determine HDF5 format version: '%s'."
                  "You may encounter unexpected errors! Good luck..."
                  % filename)
        return
    if split(version, int, np.string_('.')) < \
            split(MINIMUM_FORMAT_VERSION, int, np.string_('.')):
        raise H5VersionError("HDF5 format version {0} or newer required!\n"
                             "'{1}' has HDF5 format version {2}."
                             .format(MINIMUM_FORMAT_VERSION.decode("utf-8"),
                                     filename,
                                     version.decode("utf-8")))


class HDF5Sink(Module):
    """Write KM3NeT-formatted HDF5 files, event-by-event.

    The data can be a numpy structured array, a pandas DataFrame,
    or a km3pipe dataclass object with a `serialise()` method.

    The name of the corresponding H5 table is the decamelised
    blob-key, so values which are stored in the blob under `FooBar`
    will be written to `/foo_bar` in the HDF5 file.

    Parameters
    ----------
    filename: str, optional (default: 'dump.h5')
        Where to store the events.
    h5file: pytables.File instance, optional (default: None)
        Opened file to write to. This is mutually exclusive with filename.
    """
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename') or 'dump.h5'
        self.ext_h5file = self.get('h5file') or None
        self.pytab_file_args = self.get('pytab_file_args') or dict()
        self.indices = {}
        self._header_written = False
        # magic 10000: this is the default of the "expectedrows" arg
        # from the tables.File.create_table() function
        # at least according to the docs
        # might be able to set to `None`, I don't know...
        self.n_rows_expected = self.get('n_rows_expected') or 10000
        self.index = 0

        if self.filename != 'dump.h5' and self.ext_h5file is not None:
            raise IOError("Can't specify both filename and file object!")
        elif self.filename == 'dump.h5' and self.ext_h5file is not None:
            self.h5file = self.ext_h5file
        else:
            self.h5file = tb.open_file(self.filename, mode="w", title="KM3NeT",
                                       **self.pytab_file_args)
        self.filters = tb.Filters(complevel=5, shuffle=True, fletcher32=True,
                                  complib='zlib')
        self._tables = OrderedDict()

    def _to_array(self, data):
        if np.isscalar(data):
            return np.asarray(data).reshape((1,))
        if len(data) <= 0:
            return
        try:
            return data.to_records(index=False)
        except AttributeError:
            pass
        try:
            return data.serialise()
        except AttributeError:
            pass
        return data

    def _write_array(self, where, arr, datatype, title=''):
        level = len(where.split('/'))

        if where not in self._tables:
            dtype = arr.dtype
            loc, tabname = os.path.split(where)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', tb.NaturalNameWarning)
                tab = self.h5file.create_table(
                    loc, tabname, description=dtype, title=title,
                    filters=self.filters, createparents=True,
                    expectedrows=self.n_rows_expected,
                )
            tab._v_attrs.datatype = datatype
            if(level < 5):
                self._tables[where] = tab
        else:
            tab = self._tables[where]

        tab.append(arr)

        if(level < 4):
            tab.flush()

    def _write_separate_columns(self, where, obj, title=''):
        f = self.h5file
        loc, group_name = os.path.split(where)
        if where not in f:
            datatype = obj.__class__.__name__
            group = f.create_group(loc, group_name, datatype)
            group._v_attrs.datatype = datatype
        else:
            group = f.get_node(where)

        for col, (dt, _) in obj.dtype.fields.items():
            data = obj.__array__()[col]

            if col not in group:
                a = tb.Atom.from_dtype(dt)
                arr = f.create_earray(group, col, a, (0,), col.capitalize(),
                                      filters=self.filters)
            else:
                arr = getattr(group, col)
            arr.append(data)

        # create index table
        if where not in self.indices:
            self.indices[where] = {}
            self.indices[where]["index"] = 0
            self.indices[where]["indices"] = []
            self.indices[where]["n_items"] = []
        d = self.indices[where]
        n_items = len(obj)
        d["indices"].append(d["index"])
        d["n_items"].append(n_items)
        d["index"] += n_items

    def process(self, blob):

        if not self._header_written and "Header" in blob \
                and blob["Header"] is not None:
            header = self.h5file.create_group('/', 'header', 'Header')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', tb.NaturalNameWarning)
                for field, value in blob["Header"].items():
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    header._v_attrs[field] = value
            self._header_written = True

        for key, entry in sorted(blob.items()):
            serialisable_attributes = ('dtype', 'serialise', 'to_records')
            if any(hasattr(entry, a) for a in serialisable_attributes):
                try:
                    h5loc = entry.h5loc
                except AttributeError:
                    h5loc = '/'
                try:
                    tabname = entry.tabname
                except AttributeError:
                    tabname = decamelise(key)
                data = self._to_array(entry)
                if data is None:
                    continue
                if data.dtype.names is None:
                    dt = np.dtype((data.dtype, [(key, data.dtype)]))
                    data = data.view(dt)
                    h5loc = '/misc'
                where = os.path.join(h5loc, tabname)
                datatype = entry.__class__.__name__

                try:
                    if entry.write_separate_columns:
                        self._write_separate_columns(where, entry, title=key)
                    else:
                        self._write_array(where, data, datatype, title=key)
                except AttributeError:  # backwards compatibility
                    self._write_array(where, data, datatype, title=key)



        if not self.index % 1000:
            for tab in self._tables.values():
                tab.flush()

        self.index += 1
        return blob

    def finish(self):
        self.h5file.root._v_attrs.km3pipe = np.string_(kp.__version__)
        self.h5file.root._v_attrs.pytables = np.string_(tb.__version__)
        self.h5file.root._v_attrs.format_version = np.string_(FORMAT_VERSION)
        print("Adding index tables.")
        for where, data in self.indices.items():
            h5loc = where + "/_indices"
            print("  -> {0}".format(h5loc))
            indices = KM3DataFrame({"index": data["indices"],
                                    "n_items": data["n_items"]}, h5loc=h5loc)
            self._write_array(h5loc,
                              self._to_array(indices),
                              'Indices',
                              title="Indices")
        print("Creating pytables index tables. This may take a few minutes...")
        for tab in itervalues(self._tables):
            if 'frame_id' in tab.colnames:
                tab.cols.frame_id.create_index()
            if 'slice_id' in tab.colnames:
                tab.cols.slice_id.create_index()
            if 'dom_id' in tab.colnames:
                tab.cols.dom_id.create_index()
            if 'event_id' in tab.colnames:
                tab.cols.event_id.create_index()
            tab.flush()
        self.h5file.close()


class HDF5Pump(Pump):
    """Read KM3NeT-formatted HDF5 files, event-by-event.

    Parameters
    ----------
    filename: str
        From where to read events.
        """
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename') or None
        self.filenames = self.get('filenames') or []
        self.skip_version_check = bool(self.get('skip_version_check')) or False
        self.verbose = bool(self.get('verbose'))
        self.indices = {}
        if not self.filename and not self.filenames:
            raise ValueError("No filename(s) defined")

        if self.filename:
            self.filenames.append(self.filename)

        self.filequeue = list(self.filenames)
        self._set_next_file()

        self.event_ids = OrderedDict()
        self._n_each = OrderedDict()
        self.h5files = OrderedDict()
        for fn in self.filenames:
            # Open all files before reading any events
            # So we can raise version mismatches etc before reading anything
            if os.path.isfile(fn):
                h5file = tb.open_file(fn, 'r')
                if not self.skip_version_check:
                    check_version(h5file, fn)
            else:
                raise IOError("No such file or directory: '{0}'"
                              .format(fn))
            try:
                event_info = h5file.get_node('/', 'event_info')
                self.event_ids[fn] = event_info.cols.event_id[:]
                self._n_each[fn] = len(self.event_ids[fn])
            except tb.NoSuchNodeError:
                log.critical("No /event_info table found: '{0}'"
                             .format(fn))
                raise SystemExit
            self.h5files[fn] = h5file
        self._n_events = np.sum((v for k, v in self._n_each.items()))
        self.minmax = OrderedDict()
        n_read = 0
        for fn, n in iteritems(self._n_each):
            min = n_read
            max = n_read + n - 1
            n_read += n
            self.minmax[fn] = (min, max)
        self.index = None
        self._reset_index()


    def process(self, blob):
        try:
            blob = self.get_blob(self.index)
        except KeyError:
            self._reset_index()
            raise StopIteration
        self.index += 1
        return blob

    def _need_next(self, index):
        fname = self.current_file
        min, max = self.minmax[fname]
        return (index < min) or (index > max)

    def _set_next_file(self):
        if not self.filequeue:
            raise IndexError('No more files available!')
        self.current_file = self.filequeue.pop(0)
        if self.verbose:
            ("Reading %s..." % self.current_file)

    def _translate_index(self, fname, index):
        min, _ = self.minmax[fname]
        return index - min

    def get_blob(self, index):
        if self.index >= self._n_events:
            self._reset_index()
            raise StopIteration
        blob = Blob()
        if self._need_next(index):
            self._set_next_file()
        fname = self.current_file
        h5file = self.h5files[fname]
        evt_ids = self.event_ids[fname]
        local_index = self._translate_index(fname, index)
        event_id = evt_ids[local_index]

        # skip groups with separate columns
        # and deal with them later
        # this should be solved using hdf5 attributes in near future
        skipped_locs = []
        for tab in h5file.walk_nodes(classname="Table"):
            loc, tabname = os.path.split(tab._v_pathname)
            if loc in skipped_locs:
                continue;
            if tabname == "_indices":
                skipped_locs.append(loc)
                self.indices[loc] = h5file.get_node(loc + '/' + '_indices')
                continue
            tabname = camelise(tabname)
            try:
                dc = deserialise_map[tabname]
            except KeyError:
                dc = KM3Array
            arr = tab.read_where('event_id == %d' % event_id)
            blob[tabname] = dc.deserialise(arr, event_id=index, h5loc=loc)

        # skipped locs are now column wise datasets (usually hits)
        # currently hardcoded, in future using hdf5 attributes
        # to get the right constructor
        for loc in skipped_locs:
            idx, n_items = self.indices[loc][event_id]
            end = idx + n_items
            if loc == '/hits':
                channel_id = h5file.get_node("/hits/channel_id")[idx:end]
                dom_id = h5file.get_node("/hits/dom_id")[idx:end]
                time = h5file.get_node("/hits/time")[idx:end]
                tot = h5file.get_node("/hits/tot")[idx:end]
                triggered = h5file.get_node("/hits/triggered")[idx:end]

                datatype = h5file.get_node("/hits")._v_attrs.datatype

                if datatype == np.string_("RawHitSeries"):
                    blob["Hits"] = RawHitSeries.from_arrays(
                        channel_id, dom_id, time, tot, triggered, event_id)
                if datatype == np.string_("CRawHitSeries"):
                    pos_x = h5file.get_node("/hits/pos_x")[idx:end]
                    pos_y = h5file.get_node("/hits/pos_y")[idx:end]
                    pos_z = h5file.get_node("/hits/pos_z")[idx:end]
                    dir_x = h5file.get_node("/hits/dir_x")[idx:end]
                    dir_y = h5file.get_node("/hits/dir_y")[idx:end]
                    dir_z = h5file.get_node("/hits/dir_z")[idx:end]
                    du = h5file.get_node("/hits/du")[idx:end]
                    floor = h5file.get_node("/hits/floor")[idx:end]
                    t0s = h5file.get_node("/hits/t0")[idx:end]
                    time += t0s
                    blob["Hits"] = CRawHitSeries.from_arrays(
                        channel_id, dir_x, dir_y, dir_z, dom_id, du,
                        floor, pos_x, pos_y, pos_z, t0s, time, tot, triggered,
                        event_id)

            if loc == '/mc_hits':
                a = h5file.get_node("/mc_hits/a")[idx:end]
                origin = h5file.get_node("/mc_hits/origin")[idx:end]
                pmt_id = h5file.get_node("/mc_hits/pmt_id")[idx:end]
                time = h5file.get_node("/mc_hits/time")[idx:end]

                datatype = h5file.get_node("/mc_hits")._v_attrs.datatype

                if datatype == np.string_("McHitSeries"):
                    blob["McHits"] = McHitSeries.from_arrays(
                        a, origin, pmt_id, time, event_id)
                if datatype == np.string_("CMcHitSeries"):
                    pos_x = h5file.get_node("/mc_hits/pos_x")[idx:end]
                    pos_y = h5file.get_node("/mc_hits/pos_y")[idx:end]
                    pos_z = h5file.get_node("/mc_hits/pos_z")[idx:end]
                    dir_x = h5file.get_node("/mc_hits/dir_x")[idx:end]
                    dir_y = h5file.get_node("/mc_hits/dir_y")[idx:end]
                    dir_z = h5file.get_node("/mc_hits/dir_z")[idx:end]
                    blob["McHits"] = CMcHitSeries.from_arrays(
                        a, dir_x, dir_y, dir_z, origin, pmt_id,
                        pos_x, pos_y, pos_z, time, event_id)

        return blob

    def finish(self):
        """Clean everything up"""
        for h5 in itervalues(self.h5files):
            h5.close()

    def _reset_index(self):
        """Reset index to default value"""
        self.index = 0

    def __len__(self):
        return self._n_events

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        if self.index >= self._n_events:
            self._reset_index()
            raise StopIteration
        blob = self.get_blob(self.index)
        self.index += 1
        return blob

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_blob(index)
        elif isinstance(index, slice):
            return self._slice_generator(index)
        else:
            raise TypeError("index must be int or slice")

    def _slice_generator(self, index):
        """A simple slice generator for iterations"""
        start, stop, step = index.indices(len(self))
        for i in range(start, stop, step):
            yield self.get_blob(i)

        self.current_file = None


class H5Mono(Pump):
    """Read HDF5 files with one big table.

    Each event corresponds to a single row. Optionally, a table can have
    an index column which will be used as event id.

    A good example are the files produced by ``rootpy.root2hdf5``.

    Parameters
    ----------
    filename: str
        From where to read events.
    table: str, optional [default=None]
        Name of the table to read. If None, take the first table
        encountered in the file.
    id_col: str, optional [default=None]
        Column to use as event id. If None, simply enumerate
        the events instead.
    h5loc: str, optional [default='/']
        Path where to store when serializing to KM3HDF5.
    """
    def __init__(self, **context):
        super(H5Mono, self).__init__(**context)
        self.filename = self.require('filename')
        if os.path.isfile(self.filename):
            self.h5file = tb.open_file(self.filename, 'r')
        else:
            raise IOError("No such file or directory: '{0}'"
                          .format(self.filename))
        self.tabname = self.get('table') or None
        if self.tabname is None:
            self.table = self.h5file.list_nodes('/', classname='Table')[0]
            self.tabname = self.table.name
        self.blobkey = self.tabname
        self.h5loc = self.get('h5loc') or '/'
        self.index = None
        self._reset_index()

        self.table = self.h5file.get_node(os.path.join('/', self.tabname))
        self._n_events = self.table.shape[0]

    def get_blob(self, index):
        if self.index >= self._n_events:
            self._reset_index()
            raise StopIteration
        blob = Blob()
        arr = self.table[index]
        event_id = index
        arr = KM3Array.deserialise(arr, event_id=event_id, h5loc=self.h5loc,)
        blob[self.blobkey] = arr
        return blob

    def process(self, blob):
        try:
            blob = self.get_blob(self.index)
        except KeyError:
            self._reset_index()
            raise StopIteration
        self.index += 1
        return blob

    def finish(self):
        """Clean everything up"""
        self.h5file.close()

    def _reset_index(self):
        """Reset index to default value"""
        self.index = 0

    def __len__(self):
        return self._n_events

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        if self.index >= self._n_events:
            self._reset_index()
            raise StopIteration
        blob = self.get_blob(self.index)
        self.index += 1
        return blob

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_blob(index)
        elif isinstance(index, slice):
            return self._slice_generator(index)
        else:
            raise TypeError("index must be int or slice")

    def _slice_generator(self, index):
        """A simple slice generator for iterations"""
        start, stop, step = index.indices(len(self))
        for i in range(start, stop, step):
            yield self.get_blob(i)
