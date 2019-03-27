# Filename: hdf5.py
# pylint: disable=C0103,R0903,C901
# vim:set ts=4 sts=4 sw=4 et:
"""
Read and write KM3NeT-formatted HDF5 files.

"""
from __future__ import absolute_import, print_function, division

from collections import OrderedDict, defaultdict, namedtuple
import os.path
import warnings

import numpy as np
import tables as tb

try:
    from numba import jit
except ImportError:
    jit = lambda f: f

import km3pipe as kp
from km3pipe.core import Pump, Module, Blob
from km3pipe.dataclasses import Table, NDArray, DEFAULT_H5LOC
from km3pipe.logger import get_logger
from km3pipe.tools import decamelise, camelise, split, istype, get_jpp_revision

log = get_logger(__name__)    # pylint: disable=C0103

__author__ = "Tamas Gal and Moritz Lotze and Michael Moser"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"

FORMAT_VERSION = np.string_('5.1')
MINIMUM_FORMAT_VERSION = np.string_('4.1')


class H5VersionError(Exception):
    pass


def check_version(h5file):
    try:
        version = np.string_(h5file.root._v_attrs.format_version)
    except AttributeError:
        log.error(
            "Could not determine HDF5 format version: '%s'."
            "You may encounter unexpected errors! Good luck..." %
            h5file.filename
        )
        return
    if split(version, int, np.string_('.')) < \
            split(MINIMUM_FORMAT_VERSION, int, np.string_('.')):
        raise H5VersionError(
            "HDF5 format version {0} or newer required!\n"
            "'{1}' has HDF5 format version {2}.".format(
                MINIMUM_FORMAT_VERSION.decode("utf-8"), h5file.filename,
                version.decode("utf-8")
            )
        )


class HDF5Header(object):
    """Wrapper class for the `/raw_header` table in KM3HDF5"""

    def __init__(self, data):
        self._data = data
        self._set_attributes()

    def _set_attributes(self):
        """Traverse the internal dictionary and set the getters"""
        for parameter, data in self._data.items():
            if isinstance(data, dict) or isinstance(data, OrderedDict):
                field_names, field_values = zip(*data.items())
                sorted_indices = np.argsort(field_names)
                attr = namedtuple(
                    parameter, [field_names[i] for i in sorted_indices]
                )
                setattr(
                    self, parameter,
                    attr(*[field_values[i] for i in sorted_indices])
                )
            else:
                setattr(self, parameter, data)

    @classmethod
    def from_table(cls, table):
        data = OrderedDict()
        for i in range(len(table)):
            parameter = table['parameter'][i]
            field_names = table['field_names'][i].split(' ')
            field_values = table['field_values'][i].split(' ')
            if field_values == ['']:
                log.info(
                    "No value for parameter '{}'! Skipping...".
                    format(parameter)
                )
                continue
            dtypes = table['dtype'][i]
            dtyped_values = []
            for dtype, value in zip(dtypes.split(' '), field_values):
                if dtype.startswith('a'):
                    dtyped_values.append(value)
                else:
                    value = np.fromstring(value, dtype=dtype, sep=' ')[0]
                    dtyped_values.append(value)
            data[parameter] = OrderedDict(zip(field_names, dtyped_values))
        return cls(data)

    @classmethod
    def from_hdf5(cls, filename):
        with tb.open_file(filename, 'r') as f:
            table = f.get_node('/raw_header')
            return cls.from_pytable(table)

    @classmethod
    def from_pytable(cls, table):
        data = OrderedDict()
        for row in table:
            parameter = row['parameter'].decode()
            field_names = row['field_names'].decode().split(' ')
            field_values = row['field_values'].decode().split(' ')
            if field_values == ['']:
                log.info(
                    "No value for parameter '{}'! Skipping...".
                    format(parameter)
                )
                continue
            dtypes = row['dtype'].decode()
            dtyped_values = []
            for dtype, value in zip(dtypes.split(' '), field_values):
                if dtype.startswith('a'):
                    dtyped_values.append(value)
                else:
                    value = np.fromstring(value, dtype=dtype, sep=' ')[0]
                    dtyped_values.append(value)
            data[parameter] = OrderedDict(zip(field_names, dtyped_values))
        return cls(data)


class HDF5IndexTable(object):
    def __init__(self, h5loc):
        self.h5loc = h5loc
        self._data = defaultdict(list)
        self._index = 0

    def append(self, n_items):
        self._data['indices'].append(self._index)
        self._data['n_items'].append(n_items)
        self._index += n_items

    @property
    def data(self):
        return self._data


class HDF5Sink(Module):
    """Write KM3NeT-formatted HDF5 files, event-by-event.

    The data can be a ``kp.Table``, a numpy structured array,
    a pandas DataFrame, or a simple scalar.

    The name of the corresponding H5 table is the decamelised
    blob-key, so values which are stored in the blob under `FooBar`
    will be written to `/foo_bar` in the HDF5 file.

    Parameters
    ----------
    filename: str, optional [default: 'dump.h5']
        Where to store the events.
    h5file: pytables.File instance, optional [default: None]
        Opened file to write to. This is mutually exclusive with filename.
    complib : str [default: zlib]
        Compression library that should be used.
        'zlib', 'lzf', 'blosc' and all other PyTables filters
        are available.
    complevel : int [default: 5]
        Compression level.
    chunksize : int [optional]
        Chunksize that should be used for saving along the first axis
        of the input array.
    flush_frequency: int, optional [default: 500]
        The number of iterations to cache tables and arrays before
        dumping to disk.
    pytab_file_args: dict [optional]
        pass more arguments to the pytables File init
    n_rows_expected = int, optional [default: 10000]
    append: bool, optional [default: False]

    """

    def configure(self):
        self.filename = self.get('filename', default='dump.h5')
        self.ext_h5file = self.get('h5file')
        self.complib = self.get('complib', default='zlib')
        self.complevel = self.get('complevel', default=5)
        self.chunksize = self.get('chunksize')
        self.flush_frequency = self.get('flush_frequency', default=500)
        self.pytab_file_args = self.get('pytab_file_args', default=dict())
        self.file_mode = 'a' if self.get('append') else 'w'
        self.keep_open = self.get('keep_open')
        self.indices = {}    # to store HDF5IndexTables for each h5loc
        self._singletons_written = {}
        # magic 10000: this is the default of the "expectedrows" arg
        # from the tables.File.create_table() function
        # at least according to the docs
        # might be able to set to `None`, I don't know...
        self.n_rows_expected = self.get('n_rows_expected', default=10000)
        self.index = 0

        if self.ext_h5file is not None:
            self.h5file = self.ext_h5file
        else:
            self.h5file = tb.open_file(
                self.filename,
                mode=self.file_mode,
                title="KM3NeT",
                **self.pytab_file_args
            )
        self.filters = tb.Filters(
            complevel=self.complevel,
            shuffle=True,
            fletcher32=True,
            complib=self.complib
        )
        self._tables = OrderedDict()
        self._ndarrays = OrderedDict()
        self._ndarrays_cache = defaultdict(list)

    def _to_array(self, data, name=None):
        if data is None:
            return
        if np.isscalar(data):
            self.log.debug('toarray: is a scalar')
            return Table({name: np.asarray(data).reshape((1, ))},
                         h5loc='/misc/{}'.format(decamelise(name)),
                         name=name)
        if hasattr(data, 'len') and len(data) <= 0:    # a bit smelly ;)
            self.log.debug('toarray: data has no length')
            return
        # istype instead isinstance, to avoid heavy pandas import (hmmm...)
        if istype(data, 'DataFrame'):    # noqa
            self.log.debug('toarray: pandas dataframe')
            data = Table.from_dataframe(data)
        return data

    def _cache_ndarray(self, arr):
        self._ndarrays_cache[arr.h5loc].append(arr)

    def _write_ndarrays_cache_to_disk(self):
        """Writes all the cached NDArrays to disk and empties the cache"""
        for h5loc, arrs in self._ndarrays_cache.items():
            title = arrs[0].title
            chunkshape = (self.chunksize,) + arrs[0].shape[1:] if self.chunksize is not\
                                                           None else None

            arr = NDArray(np.concatenate(arrs), h5loc=h5loc, title=title)

            if h5loc not in self._ndarrays:
                loc, tabname = os.path.split(h5loc)
                ndarr = self.h5file.create_earray(
                    loc,
                    tabname,
                    tb.Atom.from_dtype(arr.dtype),
                    (0, ) + arr.shape[1:],
                    chunkshape=chunkshape,
                    title=title,
                    filters=self.filters,
                    createparents=True,
                )
                self._ndarrays[h5loc] = ndarr
            else:
                ndarr = self._ndarrays[h5loc]

            idx_table_h5loc = h5loc + '_indices'
            if idx_table_h5loc not in self.indices:
                self.indices[idx_table_h5loc] = HDF5IndexTable(idx_table_h5loc)
            idx_tab = self.indices[idx_table_h5loc]

            for arr_length in (len(a) for a in arrs):
                idx_tab.append(arr_length)

            ndarr.append(arr)

        self._ndarrays_cache = defaultdict(list)

    def _write_table(self, h5loc, arr, title):
        level = len(h5loc.split('/'))

        if h5loc not in self._tables:
            dtype = arr.dtype
            loc, tabname = os.path.split(h5loc)
            self.log.debug(
                "h5loc '{}', Loc '{}', tabname '{}'".format(
                    h5loc, loc, tabname
                )
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', tb.NaturalNameWarning)
                tab = self.h5file.create_table(
                    loc,
                    tabname,
                    chunkshape=self.chunksize,
                    description=dtype,
                    title=title,
                    filters=self.filters,
                    createparents=True,
                    expectedrows=self.n_rows_expected,
                )
            tab._v_attrs.datatype = title
            if (level < 5):
                self._tables[h5loc] = tab
        else:
            tab = self._tables[h5loc]

        h5_colnames = set(tab.colnames)
        tab_colnames = set(arr.dtype.names)
        if h5_colnames != tab_colnames:
            missing_cols = h5_colnames - tab_colnames
            if missing_cols:
                self.log.info(
                    "Missing columns in table, trying to append NaNs."
                )
                arr = arr.append_columns(
                    missing_cols, np.full((len(missing_cols), len(arr)),
                                          np.nan)
                )
                if (arr.dtype != tab.dtype):
                    self.log.error(
                        "Differing dtypes after appending "
                        "missing columns to the table! Skipping..."
                    )
                    return

        if arr.dtype != tab.dtype:
            try:
                arr = Table(arr, tab.dtype)
            except ValueError:
                self.log.critical(
                    "Cannot write a table to '%s' since its dtype is "
                    "different compared to the previous table with the same "
                    "HDF5 location, which was used to fix the dtype of the "
                    "HDF5 compund type." % h5loc
                )
                raise

        tab.append(arr)

        if (level < 4):
            tab.flush()

    def _write_separate_columns(self, where, obj, title):
        f = self.h5file
        loc, group_name = os.path.split(where)
        if where not in f:
            group = f.create_group(loc, group_name, title, createparents=True)
            group._v_attrs.datatype = title
        else:
            group = f.get_node(where)

        for col, (dt, _) in obj.dtype.fields.items():
            data = obj.__array__()[col]

            if col not in group:
                a = tb.Atom.from_dtype(dt)
                arr = f.create_earray(
                    group,
                    col,
                    a, (0, ),
                    col.capitalize(),
                    filters=self.filters
                )
            else:
                arr = getattr(group, col)
            arr.append(data)

        # create index table
        if where not in self.indices:
            self.indices[where] = HDF5IndexTable(where + '/_indices')
        idx_tab = self.indices[where]
        idx_tab.append(len(data))

    def _process_entry(self, key, entry):
        self.log.debug("Inspecting {}".format(key))
        if hasattr(
                entry, 'h5singleton'
        ) and entry.h5singleton and entry.h5loc in self._singletons_written:
            self.log.debug(
                "Skipping '%s' since it's a singleton and already written." %
                entry.h5loc
            )
            return
        if not hasattr(entry, 'h5loc'):
            self.log.debug("Ignoring '%s': no h5loc attribute" % key)
            return
        if isinstance(entry, NDArray):
            self._cache_ndarray(entry)
            return entry
        try:
            title = entry.name
        except AttributeError:
            title = key

        if isinstance(entry, Table) and not entry.h5singleton:
            if 'group_id' not in entry:
                entry = entry.append_columns('group_id', self.index)

        self.log.debug("h5l: '{}', title '{}'".format(entry.h5loc, title))

        if hasattr(entry, 'split_h5') and entry.split_h5:
            self.log.debug("Writing into separate columns...")
            self._write_separate_columns(entry.h5loc, entry, title=title)
        else:
            self.log.debug("Writing into single Table...")
            self._write_table(entry.h5loc, entry, title=title)

        if hasattr(entry, 'h5singleton') and entry.h5singleton:
            self._singletons_written[entry.h5loc] = True

        return entry

    def process(self, blob):
        written_blob = Blob()
        for key, entry in sorted(blob.items()):
            data = self._process_entry(key, entry)
            if data is not None:
                written_blob[key] = data

        if 'GroupInfo' not in blob:
            gi = Table(
                {
                    'group_id': self.index,
                    'blob_length': len(written_blob)
                },
                h5loc='/group_info',
                name='Group Info',
            )
            self._process_entry('GroupInfo', gi)

        if not self.index % self.flush_frequency:
            self.flush()

        self.index += 1
        return blob

    def flush(self):
        """Flush tables and arrays to disk"""
        self.log.info('Flushing tables and arrays to disk...')
        for tab in self._tables.values():
            tab.flush()
        self._write_ndarrays_cache_to_disk()

    def finish(self):
        self.flush()
        self.h5file.root._v_attrs.km3pipe = np.string_(kp.__version__)
        self.h5file.root._v_attrs.pytables = np.string_(tb.__version__)
        self.h5file.root._v_attrs.jpp = np.string_(get_jpp_revision())
        self.h5file.root._v_attrs.format_version = np.string_(FORMAT_VERSION)
        self.log.info("Adding index tables.")
        for where, idx_tab in self.indices.items():
            self.log.debug("Creating index table for '%s'" % where)
            h5loc = idx_tab.h5loc
            self.log.info("  -> {0}".format(h5loc))
            indices = Table({
                "index": idx_tab.data["indices"],
                "n_items": idx_tab.data["n_items"]
            },
                            h5loc=h5loc)
            self._write_table(h5loc, indices, title='Indices')
        self.log.info(
            "Creating pytables index tables. "
            "This may take a few minutes..."
        )
        for tab in self._tables.values():
            if 'frame_id' in tab.colnames:
                tab.cols.frame_id.create_index()
            if 'slice_id' in tab.colnames:
                tab.cols.slice_id.create_index()
            if 'dom_id' in tab.colnames:
                tab.cols.dom_id.create_index()
            if 'event_id' in tab.colnames:
                try:
                    tab.cols.event_id.create_index()
                except NotImplementedError:
                    log.warning(
                        "Table '{}' has an uint64 column, "
                        "not indexing...".format(tab._v_name)
                    )
            if 'group_id' in tab.colnames:
                try:
                    tab.cols.group_id.create_index()
                except NotImplementedError:
                    log.warning(
                        "Table '{}' has an uint64 column, "
                        "not indexing...".format(tab._v_name)
                    )
            tab.flush()

        if "HDF5MetaData" in self.services:
            self.log.info("Writing HDF5 meta data.")
            metadata = self.services["HDF5MetaData"]
            for name, value in metadata.items():
                self.h5file.set_node_attr("/", name, value)

        if not self.keep_open:
            self.h5file.close()
        self.print("HDF5 file written to: {}".format(self.filename))


class HDF5Pump(Pump):
    """Read KM3NeT-formatted HDF5 files, event-by-event.

    Parameters
    ----------
    filename: str
        From where to read events. Either this OR ``filenames`` needs to be
        defined.
    filenames: list_like(str)
        Multiple filenames. Either this OR ``filename`` needs to be defined.
    skip_version_check: bool [default: False]
        Don't check the H5 version. Might lead to unintended consequences.
    ignore_hits: bool [default: False]
        If True, do not read any hit information.
    cut_mask: str
        H5 Node path to a boolean cut mask. If specified, use the boolean array
        found at this node as a mask. ``False`` means "skip this event".
        Example: ``cut_mask="/pid/survives_precut"``
    shuffle: bool, optional [default: False]
        Shuffle the group_ids, so that the blobs are mixed up.
    shuffle_function: function, optional [default: np.random.shuffle
        The function to be used to shuffle the group IDs.
    reset_index: bool, optional [default: True]
        When shuffle is set to true, reset the group ID - start to count
        the group_id by 0.
    """

    def configure(self):
        self.filename = self.get('filename')
        self.filenames = self.get('filenames', default=[])
        self.skip_version_check = self.get('skip_version_check', default=False)
        self.verbose = bool(self.get('verbose'))
        self.ignore_hits = bool(self.get('ignore_hits'))
        self.cut_mask_node = self.get('cut_mask')
        self.shuffle = self.get('shuffle', default=False)
        self.shuffle_function = self.get(
            'shuffle_function', default=np.random.shuffle
        )
        self.reset_index = self.get('reset_index', default=False)

        self.h5file = None
        self.cut_mask = None
        self.indices = {}
        self._tab_indices = {}
        self._singletons = {}
        self.header = None
        self.group_ids = None
        self._n_groups = None
        self.index = 0

        if not self.filename and not self.filenames:
            raise ValueError("No filename(s) defined")

        if self.filename:
            self.filenames.append(self.filename)

        self.filequeue = list(self.filenames)

        self._load_next_file()

    def _load_next_file(self):
        self._reset_iteration()
        self._reset_state()
        self._open_next_file()
        if not self.skip_version_check:
            check_version(self.h5file)
        self._read_group_info()
        if self.cut_mask_node is not None:
            self._read_cut_mask()

    def _reset_state(self):
        self._close_h5file()

        self.h5file = None
        self.cut_mask = None
        self.indices = {}
        self._tab_indices = {}
        self._singletons = {}
        self.header = None
        self.group_ids = None
        self._n_groups = None
        self.index = 0

    def _open_next_file(self):
        self.log.info("Opening next file")
        if not self.filequeue:
            self.log.info("No more files available, raising StopIteration.")
            self.filequeue = list(self.filenames)
            raise StopIteration
        if self.h5file:
            self.h5file.close()
        self.filename = self.filequeue.pop(0)
        self.h5file = tb.open_file(self.filename, 'r')
        self.print("Opening {0}".format(self.filename))
        if self.verbose:
            self.print("Reading %s..." % self.filename)
        return True

    def _read_group_info(self):
        h5file = self.h5file
        if '/event_info' not in h5file and '/group_info' not in h5file:
            self.log.critical(
                "Missing /event_info or /group_info "
                "in '%s', aborting..." % h5file.filename
            )
            raise SystemExit
        elif '/group_info' in h5file:
            self.log.info("Reading group information from '/group_info'.")
            group_info = h5file.get_node('/', 'group_info')
            self.group_ids = group_info.cols.group_id[:]
            self._n_groups = len(self.group_ids)
        elif '/event_info' in h5file:
            self.log.deprecation(
                "Reading group information from '/event_info'."
            )
            event_info = h5file.get_node('/', 'event_info')
            try:
                self.group_ids = event_info.cols.group_id[:]
            except AttributeError:
                self.group_ids = event_info.cols.event_id[:]
            self._n_groups = len(self.group_ids)
        if '/raw_header' in h5file:
            self.log.info("Reading /raw_header")
            try:
                self.header = HDF5Header.from_pytable(
                    h5file.get_node('/raw_header')
                )
            except TypeError:
                self.log.error("Could not parse the raw header, skipping!")
        if self.shuffle:
            self.log.info("Shuffling group IDs")
            self.shuffle_function(self.group_ids)

    def _read_cut_mask(self):
        if not self.cut_mask_node.startswith('/'):
            self.cut_mask_node = '/' + self.cut_mask_node
        self.cut_mask = self.h5file.get_node(self.cut_mask_node)[:]
        mask = self.cut_mask
        if not mask.shape[0] == self.group_ids.shape[0]:
            raise ValueError("Cut mask length differs from event ids!")

    def process(self, blob):
        self.log.info("Reading blob at index %s" % self.index)
        if self.index >= self._n_groups:
            self.log.info("All groups are read, switching to the next file")
            if self.filequeue:
                self._load_next_file()
            else:
                self.log.info("No more files left to drain")
                raise StopIteration
        blob = self.get_blob(self.index)
        self.index += 1
        return blob

    def get_blob(self, index):
        blob = Blob()
        group_id = self.group_ids[index]
        if self.cut_mask is not None:
            self.log.debug('Cut masks found, applying...')
            mask = self.cut_mask
            if not mask[index]:
                self.log.info('Cut mask blacklists this event, skipping...')
                return

        # skip groups with separate columns
        # and deal with them later
        # this should be solved using hdf5 attributes in near future
        split_table_locs = []
        ndarray_locs = []
        for tab in self.h5file.walk_nodes(classname="Table"):
            h5loc = tab._v_pathname
            loc, tabname = os.path.split(h5loc)
            if tabname in self.indices:
                self.log.info("index table '%s' already read, skip..." % h5loc)
                continue
            if loc in split_table_locs:
                self.log.info("get_blob: '%s' is noted, skip..." % h5loc)
                continue
            if tabname == "_indices":
                self.log.debug("get_blob: found index table '%s'" % h5loc)
                split_table_locs.append(loc)
                self.indices[loc] = self.h5file.get_node(h5loc)
                continue
            if tabname.endswith("_indices"):
                self.log.debug(
                    "get_blob: found index table '%s' for NDArray" % h5loc
                )
                ndarr_loc = h5loc.replace("_indices", '')
                ndarray_locs.append(ndarr_loc)
                if ndarr_loc in self.indices:
                    self.log.info(
                        "index table for NDArray '%s' already read, skip..." %
                        ndarr_loc
                    )
                    continue
                _index_table = self.h5file.get_node(h5loc)
                self.indices[ndarr_loc] = {
                    "index": _index_table.col('index')[:],
                    "n_items": _index_table.col('n_items')[:]
                }
                continue
            tabname = camelise(tabname)

            index_column = None
            if 'group_id' in tab.dtype.names:
                index_column = 'group_id'
            elif 'event_id' in tab.dtype.names:
                index_column = 'event_id'

            if index_column is not None:
                try:
                    if h5loc not in self._tab_indices:
                        self._read_tab_indices(h5loc)
                    tab_idx_start = self._tab_indices[h5loc][0][group_id]
                    tab_n_items = self._tab_indices[h5loc][1][group_id]
                    if tab_n_items == 0:
                        continue
                    arr = tab[tab_idx_start:tab_idx_start + tab_n_items]
                except IndexError:
                    self.log.debug("No data for h5loc '%s'" % h5loc)
                    continue
                except NotImplementedError:
                    # 64-bit unsigned integer columns like ``group_id``
                    # are not yet supported in conditions
                    self.log.debug(
                        "get_blob: found uint64 column at '{}'...".
                        format(h5loc)
                    )
                    arr = tab.read()
                    arr = arr[arr[index_column] == group_id]
                except ValueError:
                    # "there are no columns taking part
                    # in condition ``group_id == 0``"
                    self.log.info(
                        "get_blob: no `%s` column found in '%s'! "
                        "skipping... " % (index_column, h5loc)
                    )
                    continue
            else:
                if h5loc not in self._singletons:
                    log.info(
                        "Caching H5 singleton: {} ({})".format(tabname, h5loc)
                    )
                    self._singletons[h5loc] = Table(
                        tab.read(),
                        h5loc=h5loc,
                        split_h5=False,
                        name=tabname,
                        h5singleton=True
                    )
                blob[tabname] = self._singletons[h5loc]
                continue

            self.log.debug("h5loc: '{}'".format(h5loc))
            tab = Table(arr, h5loc=h5loc, split_h5=False, name=tabname)
            if self.shuffle and self.reset_index:
                tab.group_id[:] = index
            blob[tabname] = tab

        # skipped locs are now column wise datasets (usually hits)
        # currently hardcoded, in future using hdf5 attributes
        # to get the right constructor
        for loc in split_table_locs:
            # if some events are missing (group_id not continuous),
            # this does not work as intended
            # idx, n_items = self.indices[loc][group_id]
            idx = self.indices[loc].col('index')[group_id]
            n_items = self.indices[loc].col('n_items')[group_id]
            end = idx + n_items
            node = self.h5file.get_node(loc)
            columns = (c for c in node._v_children if c != '_indices')
            data = {}
            for col in columns:
                data[col] = self.h5file.get_node(loc + '/' + col)[idx:end]
            tabname = camelise(loc.split('/')[-1])
            s_tab = Table(data, h5loc=loc, split_h5=True, name=tabname)
            if self.shuffle and self.reset_index:
                s_tab.group_id[:] = index
            blob[tabname] = s_tab

        if self.header is not None:
            blob['Header'] = self.header

        for ndarr_loc in ndarray_locs:
            self.log.info("Reading %s" % ndarr_loc)
            try:
                idx = self.indices[ndarr_loc]['index'][group_id]
                n_items = self.indices[ndarr_loc]['n_items'][group_id]
            except IndexError:
                continue
            end = idx + n_items
            ndarr = self.h5file.get_node(ndarr_loc)
            ndarr_name = camelise(ndarr_loc.split('/')[-1])
            _ndarr = NDArray(
                ndarr[idx:end],
                h5loc=ndarr_loc,
                title=ndarr.title,
                group_id=group_id
            )
            if self.shuffle and self.reset_index:
                _ndarr.group_id = index
            blob[ndarr_name] = _ndarr

        return blob

    def _read_tab_indices(self, h5loc):
        self.log.info("Reading table indices for '{}'".format(h5loc))
        # group_ids = self.h5file.get_node(h5loc).cols.group_id[:]
        node = self.h5file.get_node(h5loc)
        group_ids = None
        if 'group_id' in node.dtype.names:
            group_ids = self.h5file.get_node(h5loc).cols.group_id[:]
        elif 'event_id' in node.dtype.names:
            group_ids = self.h5file.get_node(h5loc).cols.event_id[:]
        else:
            self.log.error("No data found in '{}'".format(h5loc))
            return

        self._tab_indices[h5loc] = create_index_tuple(group_ids)

    def _reset_iteration(self):
        """Reset index to default value"""
        self.log.info("Resetting iteration")
        self.index = 0

    def __len__(self):
        self.log.info("Opening all HDF5 files to check the number of groups")
        n_groups = 0
        for filename in self.filenames:
            with tb.open_file(filename, 'r') as h5file:
                group_info = h5file.get_node('/', 'group_info')
                self.group_ids = group_info.cols.group_id[:]
                n_groups += len(self.group_ids)
        return n_groups

    def __iter__(self):
        return self

    def __next__(self):
        # TODO: wrap that in self._check_if_next_file_is_needed(self.index)
        if self.index >= self._n_groups:
            self.log.info("All groups are read, switching to the next file")
            if self.filequeue:
                self._load_next_file()
            else:
                self.log.info("No more files left to drain, resetting")
                self.filequeue = list(self.filenames)
                self._load_next_file()
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

        self.filename = None

    def _close_h5file(self):
        if self.h5file:
            self.h5file.close()

    def finish(self):
        self._close_h5file()


@jit
def create_index_tuple(group_ids):
    """An helper function to create index tuples for fast lookup in HDF5Pump"""
    max_group_id = np.max(group_ids)

    start_idx_arr = np.full(max_group_id + 1, 0)
    n_items_arr = np.full(max_group_id + 1, 0)

    current_group_id = group_ids[0]
    current_idx = 0
    item_count = 0

    for group_id in group_ids:
        if group_id != current_group_id:
            start_idx_arr[current_group_id] = current_idx
            n_items_arr[current_group_id] = item_count
            current_idx += item_count
            item_count = 0
            current_group_id = group_id
        item_count += 1
    else:
        start_idx_arr[current_group_id] = current_idx
        n_items_arr[current_group_id] = item_count

    return (start_idx_arr, n_items_arr)


class HDF5MetaData(Module):
    """Metadata to attach to the HDF5 file.

    Parameters
    ----------
    data: dict

    """

    def configure(self):
        self.data = self.require("data")
        self.expose(self.data, "HDF5MetaData")


def convert_header_dict_to_table(header_dict):
    """Converts a header dictionary (usually from aanet) to a Table"""
    if not header_dict:
        log.warning("Can't convert empty header dict to table, skipping...")
        return
    tab_dict = defaultdict(list)
    log.debug("Param:   field_names    field_values    dtype")
    for parameter, data in header_dict.items():
        fields = []
        values = []
        types = []
        for field_name, field_value in data.items():
            fields.append(field_name)
            values.append(str(field_value))
            try:
                _ = float(field_value)    # noqa
                types.append('f4')
            except ValueError:
                types.append('a{}'.format(len(field_value)))
        tab_dict['parameter'].append(parameter)
        tab_dict['field_names'].append(' '.join(fields))
        tab_dict['field_values'].append(' '.join(values))
        tab_dict['dtype'].append(' '.join(types))
        log.debug(
            "{}: {} {} {}".format(
                tab_dict['parameter'][-1],
                tab_dict['field_names'][-1],
                tab_dict['field_values'][-1],
                tab_dict['dtype'][-1],
            )
        )
    return Table(
        tab_dict, h5loc='/raw_header', name='RawHeader', h5singleton=True
    )
