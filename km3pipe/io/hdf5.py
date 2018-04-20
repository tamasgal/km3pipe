# Filename: hdf5.py
# pylint: disable=C0103,R0903,C901
# vim:set ts=4 sts=4 sw=4 et:
"""
Read and write KM3NeT-formatted HDF5 files.
"""
from collections import OrderedDict, defaultdict
import os.path
from six import itervalues, iteritems
import warnings

import numpy as np
import tables as tb
from pandas import DataFrame

import km3pipe as kp
from km3pipe.core import Pump, Module, Blob
from km3pipe.dataclasses import Table, DEFAULT_H5LOC
from km3pipe.dataclass_templates import TEMPLATES
from km3pipe.logger import logging
from km3pipe.tools import camelise, split

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"

FORMAT_VERSION = np.string_('5.0')
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
            self.h5file = tb.open_file(self.filename, mode=self.file_mode,
                                       title="KM3NeT",
                                       **self.pytab_file_args)
        self.filters = tb.Filters(complevel=5, shuffle=True, fletcher32=True,
                                  complib='zlib')
        self._tables = OrderedDict()

    def _to_array(self, data):
        if np.isscalar(data):
            self.log.debug('toarray: is a scalar')
            return np.asarray(data).reshape((1,))
        if len(data) <= 0:
            self.log.debug('toarray: data has no length')
            return
        if isinstance(data, DataFrame):
            self.log.debug('toarray: pandas dataframe')
            data = Table.from_dataframe(data)
        return data

    def _write_array(self, h5loc, arr, title):
        level = len(h5loc.split('/'))

        if h5loc not in self._tables:
            dtype = arr.dtype
            loc, tabname = os.path.split(h5loc)
            self.log.debug("h5loc '{}', Loc '{}', tabname '{}'".format(
                h5loc, loc, tabname))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', tb.NaturalNameWarning)
                tab = self.h5file.create_table(
                    loc, tabname, description=dtype, title=title,
                    filters=self.filters, createparents=True,
                    expectedrows=self.n_rows_expected,
                )
            tab._v_attrs.datatype = title
            if(level < 5):
                self._tables[h5loc] = tab
        else:
            tab = self._tables[h5loc]

        tab.append(arr)

        if(level < 4):
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

    def _write_header(self, header):
        header = self.h5file.create_group('/', 'header', 'Header')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', tb.NaturalNameWarning)
            for field, value in header.items():
                try:
                    value = float(value)
                except ValueError:
                    pass
                header._v_attrs[field] = value
        self._header_written = True

    def _process_entry(self, key, entry):
        self.log.debug("Inspecting {}".format(key))
        self.log.debug("Converting to numpy array...")
        data = self._to_array(entry)
        if data is None or not hasattr(data, 'dtype'):
            self.log.debug("Conversion failed. moving on...")
            return
        try:
            self.log.debug("Looking for h5loc...")
            h5loc = entry.h5loc
        except AttributeError:
            self.log.debug(
                "h5loc not found. setting to '{}'...".format(
                    DEFAULT_H5LOC))
            h5loc = DEFAULT_H5LOC
        if data.dtype.names is None:
            self.log.debug("Array has no named dtype. "
                           "using blob key as h5 column name")
            dt = np.dtype((data.dtype, [(key, data.dtype)]))
            data = data.view(dt)
        # where = os.path.join(h5loc, tabname)
        try:
            title = entry.name
        except AttributeError:
            title = key

        self.log.debug("h5l: '{}', title '{}'".format(h5loc, title))

        if hasattr(entry, 'split_h5') and entry.split_h5:
            self.log.debug("Writing into separate columns...")
            self._write_separate_columns(h5loc, entry, title=title)
        else:
            self.log.debug("Writing into single Table...")
            self._write_array(h5loc, data, title=title)

    def process(self, blob):
        if not self._header_written and "Header" in blob \
                and blob["Header"] is not None:
            self._write_header(blob['Header'])

        for key, entry in sorted(blob.items()):
            self._process_entry(key, entry)

        if not self.index % 1000:
            self.log.info('Flushing tables to disk...')
            for tab in self._tables.values():
                tab.flush()

        self.index += 1
        return blob

    def finish(self):
        self.h5file.root._v_attrs.km3pipe = np.string_(kp.__version__)
        self.h5file.root._v_attrs.pytables = np.string_(tb.__version__)
        self.h5file.root._v_attrs.format_version = np.string_(FORMAT_VERSION)
        self.log.info("Adding index tables.")
        for where, data in self.indices.items():
            h5loc = where + "/_indices"
            self.log.info("  -> {0}".format(h5loc))
            indices = Table({"index": data["indices"],
                             "n_items": data["n_items"]}, h5loc=h5loc)
            self._write_array(h5loc,
                              self._to_array(indices),
                              title='Indices',)
        self.log.info("Creating pytables index tables. "
                      "This may take a few minutes...")
        for tab in itervalues(self._tables):
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
                    log.warn("Table '{}' has an uint64 column, "
                             "not indexing...".format(tab._v_name))
            if 'group_id' in tab.colnames:
                try:
                    tab.cols.group_id.create_index()
                except NotImplementedError:
                    log.warn("Table '{}' has an uint64 column, "
                             "not indexing...".format(tab._v_name))
            tab.flush()

        if "HDF5MetaData" in self.services:
            self.log.info("Writing HDF5 meta data.")
            metadata = self.services["HDF5MetaData"]
            for name, value in metadata.items():
                self.h5file.set_node_attr("/", name, value)

        self.h5file.close()
        self.log.info("HDF5 file written to: {}".format(self.filename))


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
        if not self.filename and not self.filenames:
            raise ValueError("No filename(s) defined")

        if self.filename:
            self.filenames.append(self.filename)

        self.filequeue = list(self.filenames)
        self.h5file = None
        self._set_next_file()

        self.group_ids = OrderedDict()
        self._n_each = OrderedDict()
        for fn in self.filenames:
            self._inspect_infile(fn)
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

    def _inspect_infile(self, fn):
        # Open all files before reading any events
        # So we can raise version mismatches etc before reading anything
        self.log.debug(fn)
        if os.path.isfile(fn):
            if not self.skip_version_check:
                with tb.open_file(fn, 'r') as h5file:
                    check_version(h5file, fn)
        else:
            raise IOError("No such file or directory: '{0}'"
                          .format(fn))
        try:
            with tb.open_file(fn, 'r') as h5file:
                event_info = h5file.get_node('/', 'event_info')
            try:
                self.group_ids[fn] = event_info.cols.group_id[:]
            except AttributeError:
                self.group_ids[fn] = event_info.cols.event_id[:]
            self._n_each[fn] = len(self.group_ids[fn])
        except tb.NoSuchNodeError:
            self.log.critical("No /event_info table found: '{0}'"
                              .format(fn))
            raise SystemExit
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
        skipped_locs = []
        for tab in h5file.walk_nodes(classname="Table"):
            h5loc = tab._v_pathname
            loc, tabname = os.path.split(h5loc)
            if loc in skipped_locs:
                self.log.info(
                    "get_blob: '{}' is blacklisted, skipping...".format(
                        h5loc))
                continue
            if tabname == "_indices":
                self.log.debug(
                    "get_blob: found index table '{}'...".format(h5loc))
                skipped_locs.append(loc)
                self.indices[loc] = h5file.get_node(loc + '/' + '_indices')
                continue
            tabname = camelise(tabname)
            try:
                split_h5 = TEMPLATES[tabname]['split_h5']
            except KeyError:
                split_h5 = False
            if 'group_id' in tab.dtype.names:
                try:
                    arr = tab.read_where('group_id == %d' % group_id)
                except NotImplementedError:
                    # 64-bit unsigned integer columns like ``group_id``
                    # are not yet supported in conditions
                    self.log.debug(
                        "get_blob: found uint64 column at '{}'...".format(
                            h5loc))
                    arr = tab.read()
                    arr = arr[arr['group_id'] == group_id]
                except ValueError:
                    # "there are no columns taking part
                    # in condition ``group_id == 0``"
                    self.log.info(
                        "get_blob: no `group_id` column found in '{}'! "
                        "skipping... ".format(h5loc))
                    continue
            if 'event_id' in tab.dtype.names:
                try:
                    arr = tab.read_where('event_id == %d' % group_id)
                except NotImplementedError:
                    # 64-bit unsigned integer columns like ``event_id``
                    # are not yet supported in conditions
                    self.log.debug(
                        "get_blob: found uint64 column at '{}'...".format(
                            h5loc))
                    arr = tab.read()
                    arr = arr[arr['event_id'] == group_id]
                except ValueError:
                    # "there are no columns taking part
                    # in condition ``event_id == 0``"
                    self.log.info(
                        "get_blob: no `event_id` column found in '{}'! "
                        "skipping... ".format(h5loc))
                    continue
            self.log.debug("h5loc: '{}'".format(h5loc))
            blob[tabname] = Table(
                arr, h5loc=h5loc, split_h5=split_h5, name=tabname)

        # skipped locs are now column wise datasets (usually hits)
        # currently hardcoded, in future using hdf5 attributes
        # to get the right constructor
        for loc in skipped_locs:
            # if some events are missing (group_id not continuous),
            # this does not work as intended
            # idx, n_items = self.indices[loc][group_id]
            idx, n_items = self.indices[loc][local_index]
            end = idx + n_items
            if loc == '/hits' and not self.ignore_hits:
                channel_id = h5file.get_node("/hits/channel_id")[idx:end]
                dom_id = h5file.get_node("/hits/dom_id")[idx:end]
                time = h5file.get_node("/hits/time")[idx:end]
                tot = h5file.get_node("/hits/tot")[idx:end]
                triggered = h5file.get_node("/hits/triggered")[idx:end]

                datatype = h5file.get_node("/hits")._v_attrs.datatype

                if datatype in {np.string_("CRawHitSeries"),
                                np.string_("CalibHits")}:
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
                    blob["CalibHits"] = Table.from_template(
                        [channel_id, dir_x, dir_y, dir_z, dom_id, du,
                         floor, pos_x, pos_y, pos_z, t0s, time, tot, triggered,
                         group_id, ],
                        'CalibHits')
                else:
                    blob["Hits"] = Table.from_template(
                        [channel_id, dom_id, time, tot, triggered, group_id],
                        'Hits')

            if loc == '/mc_hits' and not self.ignore_hits:
                a = h5file.get_node("/mc_hits/a")[idx:end]
                origin = h5file.get_node("/mc_hits/origin")[idx:end]
                pmt_id = h5file.get_node("/mc_hits/pmt_id")[idx:end]
                time = h5file.get_node("/mc_hits/time")[idx:end]

                datatype = h5file.get_node("/mc_hits")._v_attrs.datatype

                if datatype in {np.string_("CMcHitSeries"),
                                np.string_("CalibMcHits")}:
                    pos_x = h5file.get_node("/mc_hits/pos_x")[idx:end]
                    pos_y = h5file.get_node("/mc_hits/pos_y")[idx:end]
                    pos_z = h5file.get_node("/mc_hits/pos_z")[idx:end]
                    dir_x = h5file.get_node("/mc_hits/dir_x")[idx:end]
                    dir_y = h5file.get_node("/mc_hits/dir_y")[idx:end]
                    dir_z = h5file.get_node("/mc_hits/dir_z")[idx:end]
                    blob["CalibMcHits"] = Table.from_template(
                        [a, dir_x, dir_y, dir_z, origin, pmt_id,
                         pos_x, pos_y, pos_z, time, group_id],
                        'CalibMcHits')
                else:
                    blob["McHits"] = Table.from_template(
                        [a, origin, pmt_id, time, group_id],
                        'McHits')

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
