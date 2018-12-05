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

import km3pipe as kp
from km3pipe.core import Pump, Module, Blob
from km3pipe.dataclasses import Table, DEFAULT_H5LOC
from km3pipe.logger import get_logger
from km3pipe.tools import decamelise, camelise, split, istype, get_jpp_revision

log = get_logger(__name__)    # pylint: disable=C0103

__author__ = "Tamas Gal and Moritz Lotze"
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


def check_version(h5file, filename):
    try:
        version = np.string_(h5file.root._v_attrs.format_version)
    except AttributeError:
        log.error(
            "Could not determine HDF5 format version: '%s'."
            "You may encounter unexpected errors! Good luck..." % filename
        )
        return
    if split(version, int, np.string_('.')) < \
            split(MINIMUM_FORMAT_VERSION, int, np.string_('.')):
        raise H5VersionError(
            "HDF5 format version {0} or newer required!\n"
            "'{1}' has HDF5 format version {2}.".format(
                MINIMUM_FORMAT_VERSION.decode("utf-8"), filename,
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
                    parameter,
                    [field_names[i] for i in sorted_indices]
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


class HDF5Sink(Module):
    """Write KM3NeT-formatted HDF5 files, event-by-event.

    The data can be a ``kp.Table``, a numpy structured array,
    a pandas DataFrame, or a simple scalar.

    The name of the corresponding H5 table is the decamelised
    blob-key, so values which are stored in the blob under `FooBar`
    will be written to `/foo_bar` in the HDF5 file.

    Parameters
    ----------
    filename: str, optional (default: 'dump.h5')
        Where to store the events.
    h5file: pytables.File instance, optional (default: None)
        Opened file to write to. This is mutually exclusive with filename.
    pytab_file_args: dict [optional]
        pass more arguments to the pytables File init
    n_rows_expected = int, optional [default: 10000]
    append: bool, optional [default: False]

    """

    def configure(self):
        self.filename = self.get('filename') or 'dump.h5'
        self.ext_h5file = self.get('h5file') or None
        self.pytab_file_args = self.get('pytab_file_args') or dict()
        self.file_mode = 'a' if self.get('append') else 'w'
        self.keep_open = self.get('keep_open')
        self.indices = {}
        self._singletons_written = {}
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
            self.h5file = tb.open_file(
                self.filename,
                mode=self.file_mode,
                title="KM3NeT",
                **self.pytab_file_args
            )
        self.filters = tb.Filters(
            complevel=5, shuffle=True, fletcher32=True, complib='zlib'
        )
        self._tables = OrderedDict()

    def _to_array(self, data, name=None):
        if data is None:
            return
        if np.isscalar(data):
            self.log.debug('toarray: is a scalar')
            return Table({
                name: np.asarray(data).reshape((1, ))
            },
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

    def _write_array(self, h5loc, arr, title):
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
        tab.append(arr)

        if (level < 4):
            tab.flush()

    def _write_separate_columns(self, where, obj, title):
        f = self.h5file
        loc, group_name = os.path.split(where)
        if where not in f:
            group = f.create_group(loc, group_name, title)
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
            self.indices[where] = {}
            self.indices[where]["index"] = 0
            self.indices[where]["indices"] = []
            self.indices[where]["n_items"] = []
        d = self.indices[where]
        n_items = len(obj)
        d["indices"].append(d["index"])
        d["n_items"].append(n_items)
        d["index"] += n_items

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
        self.log.debug("Converting to numpy array...")
        data = self._to_array(entry, name=key)
        if data is None or not hasattr(data, 'dtype'):
            self.log.debug("Conversion failed. moving on...")
            return
        try:
            self.log.debug("Looking for h5loc...")
            h5loc = entry.h5loc
        except AttributeError:
            self.log.debug(
                "h5loc not found. setting to '{}'...".format(DEFAULT_H5LOC)
            )
            h5loc = DEFAULT_H5LOC
        if data.dtype.names is None:
            self.log.debug(
                "Array has no named dtype. "
                "using blob key as h5 column name"
            )
            dt = np.dtype((data.dtype, [(key, data.dtype)]))
            data = data.view(dt)
        # where = os.path.join(h5loc, tabname)
        try:
            title = entry.name
        except AttributeError:
            title = key

        if isinstance(data, Table) and not data.h5singleton:
            if 'group_id' not in data:
                data = data.append_columns('group_id', self.index)

        # assert 'group_id' in data.dtype.names

        self.log.debug("h5l: '{}', title '{}'".format(h5loc, title))

        if hasattr(entry, 'split_h5') and entry.split_h5:
            self.log.debug("Writing into separate columns...")
            self._write_separate_columns(h5loc, entry, title=title)
        else:
            self.log.debug("Writing into single Table...")
            self._write_array(h5loc, data, title=title)

        if hasattr(entry, 'h5singleton') and entry.h5singleton:
            self._singletons_written[entry.h5loc] = True

        return data

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

        if not self.index % 1000:
            self.log.info('Flushing tables to disk...')
            for tab in self._tables.values():
                tab.flush()

        self.index += 1
        return blob

    def finish(self):
        self.h5file.root._v_attrs.km3pipe = np.string_(kp.__version__)
        self.h5file.root._v_attrs.pytables = np.string_(tb.__version__)
        self.h5file.root._v_attrs.jpp = np.string_(get_jpp_revision())
        self.h5file.root._v_attrs.format_version = np.string_(FORMAT_VERSION)
        self.log.info("Adding index tables.")
        for where, data in self.indices.items():
            h5loc = where + "/_indices"
            self.log.info("  -> {0}".format(h5loc))
            indices = Table({
                "index": data["indices"],
                "n_items": data["n_items"]
            },
                            h5loc=h5loc)
            self._write_array(
                h5loc,
                self._to_array(indices),
                title='Indices',
            )
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
    """

    def configure(self):
        self.filename = self.get('filename') or None
        self.filenames = self.get('filenames') or []
        self.skip_version_check = bool(self.get('skip_version_check')) or False
        self.verbose = bool(self.get('verbose'))
        self.ignore_hits = bool(self.get('ignore_hits'))
        self.cut_mask_node = self.get('cut_mask') or None
        self.cut_masks = defaultdict(list)
        self.indices = {}
        self._singletons = {}
        if not self.filename and not self.filenames:
            raise ValueError("No filename(s) defined")

        if self.filename:
            self.filenames.append(self.filename)

        self.filequeue = list(self.filenames)
        self.h5file = None
        self._set_next_file()

        self.headers = OrderedDict()
        self.group_ids = OrderedDict()
        self._n_each = OrderedDict()
        for fn in self.filenames:
            self._inspect_infile(fn)
        self._n_events = np.sum([v for k, v in self._n_each.items()])
        self.minmax = OrderedDict()
        n_read = 0
        for fn, n in self._n_each.items():
            min = n_read
            max = n_read + n - 1
            n_read += n
            self.minmax[fn] = (min, max)
        self.index = None
        self._reset_index()

    def _inspect_infile(self, fn):
        # Open all files before reading any events
        # So we can raise version mismatches etc before reading anything
        self.log.debug(fn)
        if os.path.isfile(fn):
            if not self.skip_version_check:
                with tb.open_file(fn, 'r') as h5file:
                    check_version(h5file, fn)
        else:
            raise IOError("No such file or directory: '{0}'".format(fn))

        self._read_group_info(fn)

        if self.cut_mask_node is not None:
            if not self.cut_mask_node.startswith('/'):
                self.cut_mask_node = '/' + self.cut_mask_node
            with tb.open_file(fn, 'r') as h5file:
                self.cut_masks[fn] = h5file.get_node(self.cut_mask_node)[:]
            self.log.debug(self.cut_masks[fn])
            mask = self.cut_masks[fn]
            if not mask.shape[0] == self.group_ids[fn].shape[0]:
                raise ValueError("Cut mask length differs from event ids!")
        else:
            self.cut_masks = None

    def _read_group_info(self, fn):
        with tb.open_file(fn, 'r') as h5file:
            if '/event_info' not in h5file and '/group_info' not in h5file:
                self.log.critical(
                    "Missing /event_info or /group_info "
                    "in '%s', aborting..." % fn
                )
                raise SystemExit
            elif '/group_info' in h5file:
                self.print("Reading group information from '/group_info'.")
                group_info = h5file.get_node('/', 'group_info')
                self.group_ids[fn] = group_info.cols.group_id[:]
                self._n_each[fn] = len(self.group_ids[fn])
            elif '/event_info' in h5file:
                self.print("Reading group information from '/event_info'.")
                event_info = h5file.get_node('/', 'event_info')
                try:
                    self.group_ids[fn] = event_info.cols.group_id[:]
                except AttributeError:
                    self.group_ids[fn] = event_info.cols.event_id[:]
                self._n_each[fn] = len(self.group_ids[fn])
            if '/raw_header' in h5file:
                try:
                    self.headers[fn] = HDF5Header.from_pytable(
                        h5file.get_node('/raw_header')
                    )
                except TypeError:
                    self.log.error("Could not parse the raw header, skipping!")

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
        if self.h5file:
            self.h5file.close()
        self.current_file = self.filequeue.pop(0)
        self.h5file = tb.open_file(self.current_file, 'r')
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
        h5file = self.h5file
        evt_ids = self.group_ids[fname]
        local_index = self._translate_index(fname, index)
        group_id = evt_ids[local_index]
        if self.cut_masks is not None:
            self.log.debug('Cut masks found, applying...')
            mask = self.cut_masks[fname]
            if not mask[local_index]:
                self.log.info('Cut mask blacklists this event, skipping...')
                return

        # skip groups with separate columns
        # and deal with them later
        # this should be solved using hdf5 attributes in near future
        split_locs = []
        for tab in h5file.walk_nodes(classname="Table"):
            h5loc = tab._v_pathname
            loc, tabname = os.path.split(h5loc)
            if loc in split_locs:
                self.log.info("get_blob: '%s' is noted, skip..." % h5loc)
                continue
            if tabname == "_indices":
                self.log.debug("get_blob: found index table '%s'" % h5loc)
                split_locs.append(loc)
                self.indices[loc] = h5file.get_node(loc + '/' + '_indices')
                continue
            tabname = camelise(tabname)

            index_column = None
            if 'group_id' in tab.dtype.names:
                index_column = 'group_id'
            elif 'event_id' in tab.dtype.names:
                index_column = 'event_id'

            if index_column is not None:
                try:
                    arr = tab.read_where('%s == %d' % (index_column, group_id))
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
                    self.print(
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
            blob[tabname] = Table(
                arr, h5loc=h5loc, split_h5=False, name=tabname
            )

        # skipped locs are now column wise datasets (usually hits)
        # currently hardcoded, in future using hdf5 attributes
        # to get the right constructor
        for loc in split_locs:
            # if some events are missing (group_id not continuous),
            # this does not work as intended
            # idx, n_items = self.indices[loc][group_id]
            idx = self.indices[loc].col('index')[local_index]
            n_items = self.indices[loc].col('n_items')[local_index]
            end = idx + n_items
            node = h5file.get_node(loc)
            columns = (c for c in node._v_children if c != '_indices')
            data = {}
            for column in columns:
                data[column] = h5file.get_node(loc + '/' + column)[idx:end]
            tabname = camelise(loc.split('/')[-1])
            blob[tabname] = Table(data, h5loc=loc, split_h5=True, name=tabname)

        if fname in self.headers:
            header = self.headers[fname]
            blob['Header'] = header

        return blob

    def _reset_index(self):
        """Reset index to default value"""
        self.index = 0

    def __len__(self):
        return self._n_events

    def __iter__(self):
        return self

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

    def finish(self):
        self.h5file.close()


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
