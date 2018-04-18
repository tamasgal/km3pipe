# Filename: evt.py
# pylint: disable=C0103,R0903
"""
Pumps for the EVT simulation dataformat.

"""
import sys

from collections import namedtuple, defaultdict

import numpy as np

from km3pipe.core import Pump, Blob
from km3pipe.logger import logging

from km3pipe.dataclasses import Table
from km3pipe.tools import unpack_nfirst
from km3pipe.mc import pdg2name, geant2pdg
from km3pipe.sys import ignored

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal, Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def try_decode_string(foo):
    try:
        return foo.decode('ascii')
    except AttributeError:
        return foo


class EvtPump(Pump):  # pylint: disable:R0902
    """Provides a pump for EVT-files.

    Parameters
    ----------
    filename: str
        The file to read the events from.
    cache_enabled: bool
        If enabled, a cache of the event indices is created when loading
        the file. Enable it if you want to jump around and inspect the
        events non-consecutively. [default: False]
    basename: str
        The common part of the filenames if you want to process multiple
        files e.g. file1.evt, file2.evt and file3.evt. During processing,
        the files will be concatenated behind the scenes.
        You need to specify the `index_stop` and `index_start`
        (1 and 3 for the example).
    suffix: str
        A string to append to each filename (before ".evt"), when basename
        is given. [default: '']
    index_start: int
        The starting index if you process multiple files at once. [default: 1]
    index_stop: int
        The last index if you process multiple files at once. [default: 1]
    n_digits: int or None
        The number of digits for indexing multiple files. [default: None]
        `None` means no leading zeros.
    exclude_tags: list of strings
        The tags in the EVT file, which should be ignored (e.g. if they
        cause parse errors)

    """

    def configure(self):
        self.filename = self.get('filename')
        self.cache_enabled = self.get('cache_enabled') or False
        self.basename = self.get('basename') or None
        self.suffix = self.get('suffix', default='')
        self.index_start = self.get('index_start', default=1)
        self.index_stop = self.get('index_stop', default=1)
        self.n_digits = self.get('n_digits', default=None)
        self.exclude_tags = self.get('exclude_tags')
        if self.exclude_tags is None:
            self.exclude_tags = []

        self.raw_header = None
        self.event_offsets = []
        self.index = 0
        self.whole_file_cached = False

        self.file_index = int(self.index_start)

        if self.basename:
            self.log.info("Got a basename ({}), constructing the first "
                          "filename.".format(self.basename))
            file_index = self._get_file_index_str()

            self.filename = "{}{}{}.evt"  \
                            .format(self.basename, file_index, self.suffix)
            self.log.info("Constructed filename: {}".format(self.filename))

        if self.filename:
            print("Opening {0}".format(self.filename))
            self.open_file(self.filename)
            self.prepare_blobs()

    def _reset(self):
        """Clear the cache."""
        self.log.info("Clearing the cache, resetting event offsets")
        self.raw_header = None
        self.event_offsets = []
        self.index = 0

    def _get_file_index_str(self):
        """Create a string out of the current file_index"""
        file_index = str(self.file_index)
        if self.n_digits is not None:
            file_index = file_index.zfill(self.n_digits)
        return file_index


    def prepare_blobs(self):
        """Populate the blobs"""
        self.raw_header = self.extract_header()
        if self.cache_enabled:
            self._cache_offsets()

    def extract_header(self):
        """Create a dictionary with the EVT header information"""
        self.log.info("Extracting the header")
        raw_header = self.raw_header = {}
        first_line = self.blob_file.readline()
        first_line = try_decode_string(first_line)
        self.blob_file.seek(0, 0)
        if not first_line.startswith(str('start_run')):
            self.log.warning("No header found.")
            return raw_header
        for line in iter(self.blob_file.readline, ''):
            line = try_decode_string(line)
            line = line.strip()
            try:
                tag, value = str(line).split(':')
            except ValueError:
                continue
            raw_header[tag] = str(value).split()
            if line.startswith(str('end_event:')):
                self._record_offset()
                return raw_header
        raise ValueError("Incomplete header, no 'end_event' tag found!")

    def get_blob(self, index):
        """Return a blob with the event at the given index"""
        self.log.info("Retrieving blob #{}".format(index))
        if index > len(self.event_offsets) - 1:
            self.log.info("Index not in cache, caching offsets")
            self._cache_offsets(index, verbose=False)
        self.blob_file.seek(self.event_offsets[index], 0)
        blob = self._create_blob()
        if blob is None:
            self.log.info("Empty blob created...")
            raise IndexError
        else:
            self.log.debug("Returning the blob")
            return blob

    def process(self, blob=None):
        """Pump the next blob to the modules"""
        try:
            blob = self.get_blob(self.index)
        except IndexError:
            self.log.info("Got an IndexError, trying the next file")
            if self.basename and self.file_index < self.index_stop:
                self.file_index += 1
                self.log.info("Now at file_index={}".format(self.file_index))
                self._reset()
                self.blob_file.close()
                self.log.info("Resetting blob index to 0")
                self.index = 0
                file_index = self._get_file_index_str()

                self.filename = "{}{}{}.evt"  \
                                .format(self.basename, file_index, self.suffix)
                self.log.info("Next filename: {}".format(self.filename))
                print("Opening {0}".format(self.filename))
                self.open_file(self.filename)
                self.prepare_blobs()
                try:
                    blob = self.get_blob(self.index)
                except IndexError:
                    self.log.warning("No blob found in file {}"
                                     .format(self.filename))
                else:
                    return blob
            self.log.info("No files left, terminating the pipeline")
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
            line = try_decode_string(line)
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
            line = try_decode_string(line)
            line = line.strip()
            if line.startswith('end_event:') and blob:
                blob['raw_header'] = self.raw_header
                return blob
            if line.startswith('start_event:'):
                blob = Blob()
                tag, value = line.split(':')
                blob[tag] = value.split()
                continue

    def __len__(self):
        if not self.whole_file_cached:
            self._cache_offsets()
        return len(self.event_offsets)

    def __iter__(self):
        return self

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

    def __init__(self, data):
        id, x, y, z, dx, dy, dz, E, t, args = unpack_nfirst(data, 9)
        self.id = int(id)
        self.pos = (x, y, z)
        self.dir = (dx, dy, dz)
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
        try:
            self.particle_type = int(self.args[0])
            self.charmed = bool(self.args[1])
            self.mother = int(self.args[2])
            self.grandmother = int(self.args[3])
            self.length = 0
        except IndexError:
            self.particle_type = geant2pdg(int(self.args[0]))
            try:
                self.length = self.args[1]
            except IndexError:
                self.length = 0

    def __repr__(self):
        text = super(self.__class__, self).__repr__()
        text += " length: {0} [m]\n".format(self.length)
        try:
            text += " type: {0} [Corsika]\n".format(self.particle_type)
            text += " charmed: {0}\n".format(self.charmed)
            text += " mother: {0} [Corsika]\n".format(self.mother)
            text += " grandmother: {0} [Corsika]\n".format(self.grandmother)
        except AttributeError:
            text += " type: {0} '{1}' [PDG]\n"  \
                    .format(self.particle_type,
                            pdg2name(self.particle_type))
            pass

        return text


class TrackCorsika(Track):
    """Representation of a track in a corsika output file"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.particle_type = int(self.args[0])
        try:
            self.charmed = bool(self.args[1])
            self.mother = int(self.args[2])
            self.grandmother = int(self.args[3])
        except IndexError:
            pass

    def __repr__(self):
        text = super(self.__class__, self).__repr__()
        text += " type: {0} [Corsika]\n".format(self.particle_type)
        try:
            text += " charmed: {0}\n".format(self.charmed)
            text += " mother: {0} [Corsika]\n".format(self.mother)
            text += " grandmother: {0} [Corsika]\n".format(self.grandmother)
        except AttributeError:
            pass
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

    def __init__(self, data):
        id, x, y, z, dx, dy, dz, E, t, Bx, By, \
            ichan, particle_type, channel, args = unpack_nfirst(data, 14)
        self.id = id
        # z correctio due to gen/km3 (ZED -> sea level shift)
        # http://wiki.km3net.physik.uni-erlangen.de/index.php/Simulations
        self.pos = (x, y, z)
        self.dir = (dx, dy, dz)
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
    return EvtRawHit(first.id, first.pmt_id, self.tot + other.tot, first.time)


EvtRawHit = namedtuple('EvtRawHit', 'id pmt_id tot time')
EvtRawHit.__new__.__defaults__ = (None, None, None, None)
EvtRawHit.__add__ = __add_raw_hit__
