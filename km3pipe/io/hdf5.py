# coding=utf-8
# Filename: hdf5.py
# pylint: disable=C0103,R0903
# vim:set ts=4 sts=4 sw=4 et:
"""
Read and write KM3NeT-formatted HDF5 files.
"""
from __future__ import division, absolute_import, print_function

from collections import OrderedDict
import os.path
from six import itervalues, iteritems
import warnings

import numpy as np
import tables as tb

import km3pipe as kp
from km3pipe.core import Pump, Module, Blob
from km3pipe.dataclasses import (KM3Array, deserialise_map, split_per_event,
        HitSeries, McHitSeries, TrackSeries, McTrackSeries)
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

FORMAT_VERSION = np.string_('4.0')
MINIMUM_FORMAT_VERSION = np.string_('4.0')


class H5VersionError(Exception):
    pass


class HDF5Sink(Module):
    """Write KM3NeT-formatted HDF5 files, event-by-event.

    The data has to have
    - a `h5loc` attribute
    - a `conv_to('numpy')` method
    To save numpy arrays/pandas dataframes, put them into a
    km3array/km3dataframe.

    The name of the corresponding H5 table is the decamelised
    blob-key, so values which are stored in the blob under `FooBar`
    will be written to `/foo_bar` in the HDF5 file.

    Parameters
    ----------
    filename: str, optional (default: 'dump.h5')
        Where to store the events.
    h5file: pytables.File instance, optional (default: None)
        Opened file to write to. This is mutually exclusive with filename.
    verbose: optional (default: False)
    """
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename') or 'dump.h5'
        self.ext_h5file = self.get('h5file') or None
        self.verbose = self.get('verbose') or False

        self.index = 0

        if self.filename != 'dump.h5' and self.ext_h5file is not None:
            raise IOError("Can't specify both filename and file object!")
        elif self.filename == 'dump.h5' and self.ext_h5file is not None:
            self.h5file = self.ext_h5file
        else:
            self.h5file = tb.open_file(self.filename, mode="w", title="KM3NeT")
        try:
            self.filters = tb.Filters(complevel=5, shuffle=True,
                                      fletcher32=True, complib='blosc')
        except tb.exceptions.FiltersWarning:
            log.error("BLOSC Compression not available, "
                      "falling back to zlib...")
            self.filters = tb.Filters(complevel=5, shuffle=True,
                                      fletcher32=True, complib='zlib')

    def _write_array(self, where, arr, title=''):
        loc, tabname = os.path.split(where)
        if where not in self.h5file:
            dtype = arr.dtype
            with warnings.catch_warnings():
                # suppress those those NaturalNameWarnigns
                # because we have integes like `/hits/0`
                warnings.simplefilter("ignore")
                tab = self.h5file.create_table(loc, tabname,
                        description=dtype, title=title, filters=self.filters,
                        createparents=True)
        else:
            tab = self.h5file.get_node(where)
        tab.append(arr)

    def process(self, blob):
        for key, entry in sorted(iteritems(blob)):
            if not (hasattr(entry, 'h5loc') and hasattr(entry, 'conv_to')):
                continue
            try:
                tabname = entry.tabname
            except AttributeError:
                tabname = decamelise(key)
            entry = entry.conv_to('numpy')
            if entry.dtype.names is None:
                dt = np.dtype((entry.dtype, [(key, entry.dtype)]))
                entry = entry.view(dt)
            h5loc = entry.h5loc
            if tabname in split_per_event:
                where = '{}/{}'.format(h5loc, self.index)
            else:
                where = os.path.join(h5loc, tabname)
            self._write_array(where, entry, title=key)

        self.index += 1
        return blob

    def finish(self):
        self.h5file.root._v_attrs.km3pipe = np.string_(kp.__version__)
        self.h5file.root._v_attrs.pytables = np.string_(tb.__version__)
        self.h5file.root._v_attrs.format_version = np.string_(FORMAT_VERSION)
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
        if not self.filename and not self.filenames:
            raise ValueError("No filename(s) defined")

        if self.filename:
            self.filenames.append(self.filename)

        self._filequeue = list(self.filenames)
        self._set_next_file()

        self._n_events = OrderedDict()
        self._h5files = OrderedDict()
        for fn in self.filenames:
            # Open all files before reading any events
            # So we can raise version mismatches etc before reading anything
            if os.path.isfile(fn):
                h5file = tb.open_file(fn, 'r')
                if not self.skip_version_check:
                    self._check_version(h5file, fn)
            else:
                raise IOError("No such file or directory: '{0}'"
                              .format(fn))
            try:
                event_info = h5file.get_node('/', 'event_info')
                self._n_events[fn] = event_info.shape[0]
            except tb.NoSuchNodeError:
                log.critical("No /event_info table found: '{0}'"
                             .format(fn))
                raise SystemExit
            self._check_available_tables(h5file)
            self._h5files[fn] = h5file
        self._n_events_total = np.sum((v for k, v in iteritems(self._n_events)))
        self._minmax = OrderedDict()
        n_read = 0
        for fn, n in iteritems(self._n_events):
            min = n_read
            max = n_read + n - 1
            n_read += n
            self._minmax[fn] = (min, max)
        self.index = None
        self._reset_index()


    def _check_available_tables(self, h5):
        self._has_hits = False
        self._has_mc_hits = False
        self._has_tracks = False
        self._has_mc_tracks = False
        self._ordinary_tabs = set()
        # so far we ignore slices/frames
        for leaf in h5.walk_nodes(classname='Leaf'):
            if '/hits' in leaf._v_pathname:
                if not self._has_hits:
                    self._has_hits = True
                continue
            if '/mc_hits' in leaf._v_pathname:
                if not self._has_mc_hits:
                    self._has_mc_hits = True
                continue
            if '/tracks' in leaf._v_pathname:
                if not self._has_tracks:
                    self._has_tracks = True
                continue
            if '/mc_tracks' in leaf._v_pathname:
                if not self._has_mc_tracks:
                    self._has_mc_tracks = True
                continue
            self._ordinary_tabs.add(leaf._v_pathname)

    def _check_version(self, h5file, filename):
        try:
            version = np.string_(h5file.root._v_attrs.format_version)
        except AttributeError:
            log.error("Could not determine HDF5 format version: '%s'."
                      "You may encounter unexpected errors! Good luck..."
                      % filename)
            return
        if split(version, int, np.string_('.')) < \
                split(MINIMUM_FORMAT_VERSION, int, np.string_('.')):
            raise H5VersionError(
                    "HDF5 format version {0} or newer required!\n"
                    "'{1}' has HDF5 format version {2}." .format(
                        MINIMUM_FORMAT_VERSION, filename, version))

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
        min, max = self._minmax[fname]
        return (index < min) or (index > max)

    def _set_next_file(self):
        if not self._filequeue:
            raise IndexError('No more files available!')
        self.current_file = self._filequeue.pop(0)
        if self.verbose:
            ("Reading %s..." % self.current_file)

    def _translate_index(self, fname, index):
        min, _ = self._minmax[fname]
        return index - min

    def _is_split_per_event(self, where):
        for k in split_per_event:
            if k in where:
                return True
        return False

    def get_blob(self, index):
        if self.index >= self._n_events_total:
            self._reset_index()
            raise StopIteration
        blob = Blob()
        if self._need_next(index):
            self._set_next_file()
        fname = self.current_file
        h5file = self._h5files[fname]
        n_events = self._n_events[fname]
        local_index = self._translate_index(fname, index)

        if self._has_hits:
            hits = h5file.get_node('/hits/{}'.format(local_index))[:]
            blob['Hits'] = HitSeries.conv_from(hits)
        if self._has_mc_hits:
            mc_hits = h5file.get_node('/mc_hits/{}'.format(local_index))[:]
            blob['McHits'] = McHitSeries.conv_from(mc_hits)
        if self._has_tracks:
            tracks = h5file.get_node('/tracks/{}'.format(local_index))[:]
            blob['Tracks'] = TrackSeries.conv_from(tracks)
        if self._has_mc_tracks:
            mc_tracks = h5file.get_node('/mc_tracks/{}'.format(local_index))[:]
            blob['McTracks'] = McTrackSeries.conv_from(mc_tracks)

        for where in self._ordinary_tabs:
            tab = h5file.get_node(where)
            loc, tabname = os.path.split(where)
            tabname = camelise(tabname)
            #arr = tab.read_where('event_id == %d' % event_id)
            arr = np.atleast_1d(tab[local_index])
            blob[tabname] = KM3Array.conv_from(arr, h5loc=loc)
        return blob

    def finish(self):
        """Clean everything up"""
        for h5 in itervalues(self.h5files):
            h5.close()

    def _reset_index(self):
        """Reset index to default value"""
        self.index = 0

    def __len__(self):
        return self._n_events_total

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        if self.index >= self._n_events_total:
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
