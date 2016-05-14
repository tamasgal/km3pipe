# coding=utf-8
# Filename: evt.py
# pylint: disable=C0103,R0903
"""
Pumps for the EVT simulation dataformat.

"""
from __future__ import division, absolute_import, print_function

import sys

from collections import namedtuple

from km3pipe import Pump
from km3pipe.logger import logging

from km3pipe.dataclasses import Point, Direction, HitSeries
from km3pipe.tools import pdg2name, geant2pdg, unpack_nfirst, ignored

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = 'tamasgal'


class EvtPump(Pump):  # pylint: disable:R0902
    """Provides a pump for EVT-files"""

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename')
        self.cache_enabled = self.get('cache_enabled') or False
        self.basename = self.get('basename') or None
        self.index_start = self.get('index_start') or 1
        self.index_stop = self.get('index_stop') or 1

        self.raw_header = None
        self.event_offsets = []
        self.index = 0
        self.whole_file_cached = False

        self.file_index = int(self.index_start)

        if self.basename:
            self.filename = self.basename + str(self.index_start) + '.evt'

        if self.filename:
            print("Opening {0}".format(self.filename))
            self.open_file(self.filename)
            self.prepare_blobs()

    def _reset(self):
        """Clear the cache."""
        self.raw_header = None
        self.event_offsets = []
        self.index = 0

    def prepare_blobs(self):
        """Populate the blobs"""
        self.raw_header = self.extract_header()
        if self.cache_enabled:
            self._cache_offsets()

    def extract_header(self):
        """Create a dictionary with the EVT header information"""
        raw_header = self.raw_header = {}
        first_line = self.blob_file.readline()
        self.blob_file.seek(0, 0)
        if not first_line.startswith('start_run'):
            log.warning("No header found.")
            return raw_header
        for line in iter(self.blob_file.readline, ''):
            line = line.strip()
            try:
                tag, value = line.split(':')
            except ValueError:
                continue
            raw_header[tag] = value.split()
            if line.startswith('end_event:'):
                self._record_offset()
                return raw_header
        raise ValueError("Incomplete header, no 'end_event' tag found!")

    def get_blob(self, index):
        """Return a blob with the event at the given index"""
        if index > len(self.event_offsets) - 1:
            self._cache_offsets(index, verbose=False)
        self.blob_file.seek(self.event_offsets[index], 0)
        blob = self._create_blob()
        if blob is None:
            raise IndexError
        else:
            return blob

    def process(self, blob=None):
        """Pump the next blob to the modules"""
        try:
            blob = self.get_blob(self.index)
        except IndexError:
            if self.basename and self.file_index < self.index_stop:
                self.file_index += 1
                self._reset()
                self.blob_file.close()
                self.index = 0
                self.filename = self.basename + str(self.file_index) + '.evt'
                print("Opening {0}".format(self.filename))
                self.open_file(self.filename)
                self.prepare_blobs()
                return blob
            raise StopIteration
        self.index += 1
        return blob

    def _cache_offsets(self, up_to_index=None, verbose=True):
        """Cache all event offsets."""
        if not up_to_index:
            if verbose:
                print("Caching event file offsets, this may take a minute.")
            self.blob_file.seek(0, 0)
            self.event_offsets = []
            if not self.raw_header:
                self.event_offsets.append(0)
        else:
            self.blob_file.seek(self.event_offsets[-1], 0)
        for line in iter(self.blob_file.readline, ''):
            line = line.strip()
            if line.startswith('end_event:'):
                self._record_offset()
                if len(self.event_offsets) % 100 == 0:
                    if verbose:
                        print('.', end='')
                    sys.stdout.flush()
            if up_to_index and len(self.event_offsets) >= up_to_index + 1:
                return
        self.event_offsets.pop()  # get rid of the last entry
        if not up_to_index:
            self.whole_file_cached = True
        print("\n{0} events indexed.".format(len(self.event_offsets)))

    def _record_offset(self):
        """Stores the current file pointer position"""
        offset = self.blob_file.tell()
        self.event_offsets.append(offset)

    def _create_blob(self):
        """Parse the next event from the current file position"""
        blob = None
        for line in self.blob_file:
            line = line.strip()
            if line.startswith('end_event:') and blob:
                blob['raw_header'] = self.raw_header
                with ignored(KeyError):
                    blob['Hits'] = HitSeries.from_evt(blob['EvtRawHits'],
                                                      self.index)
                return blob
            if line.startswith('start_event:'):
                blob = {}
                tag, value = line.split(':')
                blob[tag] = value.split()
                continue
            if blob:
                self._create_blob_entry_for_line(line, blob)

    def _create_blob_entry_for_line(self, line, blob):
        """Create the actual blob entry from the given line."""
        try:
            tag, value = line.split(':')
        except ValueError:
            log.warning("Corrupt line in EVT file:\n{0}".format(line))
            return
        if tag in ('track_in', 'track_fit', 'hit', 'hit_raw'):
            values = [float(x) for x in value.split()]
            blob.setdefault(tag, []).append(values)
            if tag == 'hit':
                hit = EvtHit(*values)
                blob.setdefault("EvtHits", []).append(hit)
                blob.setdefault("MCHits", []).append(hit)
            if tag == "hit_raw":
                raw_hit = EvtRawHit(*values)
                blob.setdefault("EvtRawHits", []).append(raw_hit)
            if tag == "track_in":
                blob.setdefault("TrackIns", []).append(TrackIn(values))
            if tag == "track_fit":
                blob.setdefault("TrackFits", []).append(TrackFit(values))
        else:
            if tag == 'neutrino':
                values = [float(x) for x in value.split()]
                blob['Neutrino'] = Neutrino(values)
            else:
                blob[tag] = value.split()

    def __len__(self):
        if not self.whole_file_cached:
            self._cache_offsets()
        return len(self.event_offsets)

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        try:
            blob = self.get_blob(self.index)
        except IndexError:
            self.index = 0
            raise StopIteration
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

    def finish(self):
        """Clean everything up"""
        self.blob_file.close()


class Track(object):
    """Bass class for particle or shower tracks"""
#    def __init__(self, id, x, y, z, dx, dy, dz, E=None, t=0, *args):
    def __init__(self, data, zed_correction=405.93):
        id, x, y, z, dx, dy, dz, E, t, args = unpack_nfirst(data, 9)
        self.id = int(id)
        # z correctio due to gen/km3 (ZED -> sea level shift)
        # http://wiki.km3net.physik.uni-erlangen.de/index.php/Simulations
        self.pos = Point((x, y, z + zed_correction))
        self.dir = Direction((dx, dy, dz))
        self.E = E
        self.time = t
        self.args = args

    def __repr__(self):
        text = "Track:\n"
        text += " id: {0}\n".format(self.id)
        text += " pos: {0}\n".format(self.pos)
        text += " dir: {0}\n".format(self.dir)
        text += " energy: {0} GeV\n".format(self.E)
        text += " time: {0} ns\n".format(self.time)
        return text


class TrackIn(Track):
    """Representation of a track_in entry in an EVT file"""
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.particle_type = geant2pdg(int(self.args[0]))
        try:
            self.length = self.args[1]
        except IndexError:
            self.length = 0

    def __repr__(self):
        text = super(self.__class__, self).__repr__()
        text += " type: {0} '{1}' [PDG]\n".format(self.particle_type,
                                                  pdg2name(self.particle_type))
        text += " length: {0} [m]\n".format(self.length)
        return text


class TrackFit(Track):
    """Representation of a track_fit entry in an EVT file"""
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.speed = self.args[0]
        self.ts = self.args[1]
        self.te = self.args[2]
        self.con1 = self.args[3]
        self.con2 = self.args[4]

    def __repr__(self):
        text = super(self.__class__, self).__repr__()
        text += " speed: {0} [m/ns]\n".format(self.speed)
        text += " ts: {0} [ns]\n".format(self.ts)
        text += " te: {0} [ns]\n".format(self.te)
        text += " con1: {0}\n".format(self.con1)
        text += " con2: {0}\n".format(self.con2)
        return text


class Neutrino(object):  # pylint: disable:R0902
    """Representation of a neutrino entry in an EVT file"""
    def __init__(self, data, zed_correction=405.93):
        id, x, y, z, dx, dy, dz, E, t, Bx, By, \
            ichan, particle_type, channel, args = unpack_nfirst(data, 14)
        self.id = id
        # z correctio due to gen/km3 (ZED -> sea level shift)
        # http://wiki.km3net.physik.uni-erlangen.de/index.php/Simulations
        self.pos = Point((x, y, z + zed_correction))
        self.dir = Direction((dx, dy, dz))
        self.E = E
        self.time = t
        self.Bx = Bx
        self.By = By
        self.ichan = ichan
        self.particle_type = particle_type
        self.channel = channel

    def __str__(self):
        text = "Neutrino: "
        text += pdg2name(self.particle_type)
        if self.E >= 1000000:
            text += ", {0:.3} PeV".format(self.E / 1000000)
        elif self.E >= 1000:
            text += ", {0:.3} TeV".format(self.E / 1000)
        else:
            text += ", {0:.3} GeV".format(float(self.E))
        text += ', CC' if int(self.channel) == 2 else ', NC'
        return text


# The hit entry in an EVT file
EvtHit = namedtuple('EvtHit',
                    'id pmt_id pe time type n_photons track_in c_time')
EvtHit.__new__.__defaults__ = (None, None, None, None, None, None, None, None)


# The hit_raw entry in an EVT file
def __add_raw_hit__(self, other):
    """Add two hits by adding the ToT and preserve time and pmt_id
    of the earlier one."""
    first = self if self.time <= other.time else other
    return EvtRawHit(first.id, first.pmt_id, self.tot+other.tot, first.time)
EvtRawHit = namedtuple('EvtRawHit', 'id pmt_id tot time')
EvtRawHit.__new__.__defaults__ = (None, None, None, None)
EvtRawHit.__add__ = __add_raw_hit__
