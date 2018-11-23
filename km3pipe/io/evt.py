# Filename: evt.py
# pylint: disable=C0103,R0903
"""
Pumps for the EVT simulation dataformat.

"""
from __future__ import absolute_import, print_function, division

from collections import defaultdict
import sys

import numpy as np

from km3pipe.core import Pump, Blob
from km3pipe.dataclasses import Table
from km3pipe.logger import get_logger
from km3pipe.tools import split

log = get_logger(__name__)    # pylint: disable=C0103

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal, Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def try_decode_string(text):
    """Decode string to ASCII if possible"""
    try:
        return text.decode('ascii')
    except AttributeError:
        return text


class EvtPump(Pump):    # pylint: disable:R0902
    """Provides a pump for EVT-files.

    Parameters
    ----------
    filename: str
        The file to read the events from.
    parsers: list of str or callables
        The parsers to apply for each blob (e.g. parsers=['km3sim', a_parser])
        You can also pass your own function, which takes a single argument
        `blob` and mutates it. `str` values will be looked up in the
        `kp.io.evt.EVT_PARSERS` dictionary and ignored if not found.
        If `parsers='auto'`, the `EvtPump` will try to find the appropriate
        parsers, which is the default behaviour. [default: 'auto']
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
        self.filename = self.get('filename', default=None)
        self.filenames = self.get('filenames', default=[])
        parsers = self.get('parsers', default='auto')
        self.cache_enabled = self.get('cache_enabled', default=False)
        self.basename = self.get('basename', default=None)
        self.suffix = self.get('suffix', default='')
        self.index_start = self.get('index_start', default=1)
        if self.filenames:
            self.filename = self.filenames[0]
            self.index_stop = len(self.filenames)
        else:
            self.index_stop = self.get('index_stop', default=1)
        self.n_digits = self.get('n_digits', default=None)
        self.exclude_tags = self.get('exclude_tags')
        if self.exclude_tags is None:
            self.exclude_tags = []

        self.raw_header = None
        self.event_offsets = []
        self.index = 0
        self.whole_file_cached = False
        self.parsers = []
        self._auto_parse = False

        if not self.filename and not self.filenames and not self.basename:
            print("No file- or basename(s) defined!")

        if parsers:
            if parsers == 'auto':
                self.print("Automatic tag parsing enabled.")
                self._auto_parse = True
            else:
                if isinstance(parsers, str):
                    # expects a list(str)
                    parsers = [parsers]
                self._register_parsers(parsers)

        self.file_index = int(self.index_start)
        if self.filenames:
            self.filename = self.filenames[self.file_index - 1]
        elif self.basename:
            self.log.info(
                "Got a basename ({}), constructing the first "
                "filename.".format(self.basename)
            )
            file_index = self._get_file_index_str()

            self.filename = "{}{}{}.evt"  \
                            .format(self.basename, file_index, self.suffix)
            self.log.info("Constructed filename: {}".format(self.filename))

        if self.filename:
            self.print("Opening {0}".format(self.filename))
            self.open_file(self.filename)
            self.prepare_blobs()

    def _register_parsers(self, parsers):
        self.log.info("Found parsers {}".format(parsers))
        for parser in parsers:
            if callable(parser):
                self.parsers.append(parser)
                continue

            if parser in EVT_PARSERS.keys():
                self.parsers.append(EVT_PARSERS[parser])
            else:
                self.log.warning(
                    "Parser '{}' not found, ignoring...".format(parser)
                )

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
        raw_header = self.raw_header = defaultdict(list)
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
            raw_header[tag].append(str(value).split())
            if line.startswith(str('end_event:')):
                self._record_offset()
                if self._auto_parse and 'physics' in raw_header:
                    parsers = [p[0].lower() for p in raw_header['physics']]
                    self._register_parsers(parsers)
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
            self.log.debug("Applying parsers...")
            for parser in self.parsers:
                parser(blob)
            self.log.debug("Returning the blob")
            return blob

    def process(self, blob=None):
        """Pump the next blob to the modules"""
        try:
            blob = self.get_blob(self.index)

        except IndexError:
            self.log.info("Got an IndexError, trying the next file")
            if (self.basename
                    or self.filenames) and self.file_index < self.index_stop:
                self.file_index += 1
                self.log.info("Now at file_index={}".format(self.file_index))
                self._reset()
                self.blob_file.close()
                self.log.info("Resetting blob index to 0")
                self.index = 0
                file_index = self._get_file_index_str()
                if self.filenames:
                    self.filename = self.filenames[self.file_index - 1]
                elif self.basename:
                    self.filename = "{}{}{}.evt"  \
                                    .format(self.basename, file_index, self.suffix)
                self.log.info("Next filename: {}".format(self.filename))
                self.print("Opening {0}".format(self.filename))
                self.open_file(self.filename)
                self.prepare_blobs()
                try:
                    blob = self.get_blob(self.index)
                except IndexError:
                    self.log.warning(
                        "No blob found in file {}".format(self.filename)
                    )
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
                self.print("Caching event file offsets, this may take a bit.")
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
        self.event_offsets.pop()    # get rid of the last entry
        if not up_to_index:
            self.whole_file_cached = True
        self.print("\n{0} events indexed.".format(len(self.event_offsets)))

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
            if line == '':
                self.log.info("Ignoring empty line...")
                continue
            if line.startswith('end_event:') and blob:
                blob['raw_header'] = self.raw_header
                return blob
            try:
                tag, values = line.split(':')
            except ValueError:
                self.log.warning("Ignoring corrupt line: {}".format(line))
                continue
            try:
                values = tuple(split(values.strip(), callback=float))
            except ValueError:
                self.log.info("Empty value: {}".format(values))
            if line.startswith('start_event:'):
                blob = Blob()
                blob[tag] = tuple(int(v) for v in values)
                continue
            if tag not in blob:
                blob[tag] = []
            blob[tag].append(values)

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


class Parser(object):
    """Standard parser to create numpy times from EVT raw data.

    The `tag_description` is a dict of tuples. The key is the target blob-key,
    the value is tuple of "target blob-key" and "numpy dtype".

    """

    def __init__(self, tag_description):
        self.tag_description = tag_description

    def __call__(self, blob):
        """Iterate through the blob-keys and add the parsed data to the blob"""
        for key in list(blob.keys()):
            if key in self.tag_description.keys():
                data = blob[key]
                out_key, dtype = self.tag_description[key]
                arr = np.array(data, dtype)
                tab = Table(arr, name=out_key)
                blob[out_key] = tab


KM3SIM_TAGS = {
    'hit': [
        'KM3SimHits',
        [
            ('id', 'f4'),
            ('pmt_id', '<i4'),
            ('pe', 'f4'),
            ('time', 'f4'),
            ('type', 'f4'),
            ('n_photons', 'f4'),
            ('track_in', 'f4'),
            ('c_time', 'f4'),
            ('unknown', 'f4'),
        ],
    ],
}

GSEAGEN_TAGS = {
    'neutrino': [
        'Neutrinos',
        [
            ('id', '<i4'),
            ('pos_x', 'f4'),
            ('pos_y', 'f4'),
            ('pos_z', 'f4'),
            ('dir_x', 'f4'),
            ('dir_y', 'f4'),
            ('dir_z', 'f4'),
            ('energy', 'f4'),
            ('time', 'f4'),
            ('bjorken_x', 'f4'),
            ('bjorken_y', 'f4'),
            ('scattering_type', '<i4'),
            ('pdg_id', '<i4'),
            ('interaction_type', '<i4'),
        ]
    ],
    'track_in': [
        'TrackIns',
        [
            ('id', '<i4'),
            ('pos_x', 'f4'),
            ('pos_y', 'f4'),
            ('pos_z', 'f4'),
            ('dir_x', 'f4'),
            ('dir_y', 'f4'),
            ('dir_z', 'f4'),
            ('energy', 'f4'),
            ('time', 'f4'),
            ('geant_id', 'f4'),
        ]
    ],
    'primary_lepton': [
        'PrimaryLeptons',
        [
            ('id', '<i4'),
            ('pos_x', 'f4'),
            ('pos_y', 'f4'),
            ('pos_z', 'f4'),
            ('dir_x', 'f4'),
            ('dir_y', 'f4'),
            ('dir_z', 'f4'),
            ('energy', 'f4'),
            ('time', 'f4'),
            ('geant_id', 'f4'),
        ]
    ]
}

KM3_TAGS = {
    'neutrino': [
        'Neutrinos',
        [
            ('id', '<i4'),
            ('pos_x', 'f4'),
            ('pos_y', 'f4'),
            ('pos_z', 'f4'),
            ('dir_x', 'f4'),
            ('dir_y', 'f4'),
            ('dir_z', 'f4'),
            ('energy', 'f4'),
            ('time', 'f4'),
            ('bjorken_x', 'f4'),
            ('bjorken_y', 'f4'),
            ('scattering_type', '<i4'),
            ('pdg_id', '<i4'),
            ('interaction_type', '<i4'),
        ]
    ],
    'track_in': [
        'TrackIns',
        [
            ('id', '<i4'),
            ('pos_x', 'f4'),
            ('pos_y', 'f4'),
            ('pos_z', 'f4'),
            ('dir_x', 'f4'),
            ('dir_y', 'f4'),
            ('dir_z', 'f4'),
            ('energy', 'f4'),
            ('time', 'f4'),
            ('type', 'f4'),
            ('something', '<i4'),
        ]
    ],
    'hit_raw': [
        'Hits',
        [
            ('id', '<i4'),
            ('pmt_id', '<i4'),
            ('npe', '<i4'),
            ('time', 'f4'),
        ]
    ],
}


def parse_corant(blob):
    """Creates new blob entries for the given blob keys"""

    if 'track_seamuon' in blob.keys():

        muon = blob['track_seamuon']

        blob['Muon'] = Table({
            'id': np.array(muon)[:, 0].astype(int),
            'pos_x': np.array(muon)[:, 1],
            'pos_y': np.array(muon)[:, 2],
            'pos_z': np.array(muon)[:, 3],
            'dir_x': np.array(muon)[:, 4],
            'dir_y': np.array(muon)[:, 5],
            'dir_z': np.array(muon)[:, 6],
            'energy': np.array(muon)[:, 7],
            'time': np.array(muon)[:, 8],
            'particle_id': np.array(muon)[:, 9].astype(int),
            'is_charm': np.array(muon)[:, 10].astype(int),
            'mother_pid': np.array(muon)[:, 11].astype(int),
            'grandmother_pid': np.array(muon)[:, 11].astype(int),
        },
                             h5loc='muon')

        blob['MuonMultiplicity'] = Table({
            'muon_multiplicity': len(np.array(muon)[:, 6])
        },
                                         h5loc='muon_multiplicity')

    if 'track_seaneutrino' in blob.keys():

        nu = blob['track_seaneutrino']

        blob['Neutrino'] = Table({
            'id': np.array(nu)[:, 0].astype(int),
            'pos_x': np.array(nu)[:, 1],
            'pos_y': np.array(nu)[:, 2],
            'pos_z': np.array(nu)[:, 3],
            'dir_x': np.array(nu)[:, 4],
            'dir_y': np.array(nu)[:, 5],
            'dir_z': np.array(nu)[:, 6],
            'energy': np.array(nu)[:, 7],
            'time': np.array(nu)[:, 8],
            'particle_id': np.array(nu)[:, 9].astype(int),
            'is_charm': np.array(nu)[:, 10].astype(int),
            'mother_pid': np.array(nu)[:, 11].astype(int),
            'grandmother_pid': np.array(nu)[:, 11].astype(int),
        },
                                 h5loc='nu')
        blob['NeutrinoMultiplicity'] = Table({
            'total': len(np.array(nu)[:, 6]),
            'nue': len(np.array(nu)[:, 6][np.array(nu)[:, 9] == 66]),
            'anue': len(np.array(nu)[:, 6][np.array(nu)[:, 9] == 67]),
            'numu': len(np.array(nu)[:, 6][np.array(nu)[:, 9] == 68]),
            'anumu': len(np.array(nu)[:, 6][np.array(nu)[:, 9] == 69]),
        },
                                             h5loc='nu_multiplicity')

    if ('track_seamuon' or 'track_seaneutrino') in blob.keys():

        blob['Weights'] = Table({
            'w1': blob['weights'][0][0],
            'w2': blob['weights'][0][1],
            'w3': blob['weights'][0][2],
        },
                                h5loc='weights')

    if 'track_primary' in blob.keys():

        primary = blob['track_primary']

        blob['Primary'] = Table({
            'id': np.array(primary)[:, 0].astype(int),
            'pos_x': np.array(primary)[:, 1],
            'pos_y': np.array(primary)[:, 2],
            'pos_z': np.array(primary)[:, 3],
            'dir_x': np.array(primary)[:, 4],
            'dir_y': np.array(primary)[:, 5],
            'dir_z': np.array(primary)[:, 6],
            'energy': np.array(primary)[:, 7],
            'time': np.array(primary)[:, 8],
            'particle_id': np.array(primary)[:, 9].astype(int)
        },
                                h5loc='primary')

    return blob


def parse_propa(blob):
    """Creates new blob entries for the given blob keys"""

    if 'track_in' in blob.keys():

        muon = blob['track_in']

        blob['Muon'] = Table({
            'id': np.array(muon)[:, 0].astype(int),
            'pos_x': np.array(muon)[:, 1],
            'pos_y': np.array(muon)[:, 2],
            'pos_z': np.array(muon)[:, 3],
            'dir_x': np.array(muon)[:, 4],
            'dir_y': np.array(muon)[:, 5],
            'dir_z': np.array(muon)[:, 6],
            'energy': np.array(muon)[:, 7],
            'time': np.array(muon)[:, 8],
            'particle_id': np.array(muon)[:, 9].astype(int),
            'is_charm': np.array(muon)[:, 10].astype(int),
            'mother_pid': np.array(muon)[:, 11].astype(int),
            'grandmother_pid': np.array(muon)[:, 11].astype(int),
        },
                             h5loc='muon')

        blob['MuonMultiplicity'] = Table({
            'muon_multiplicity': len(np.array(muon)[:, 6])
        },
                                         h5loc='muon_multiplicity')

    if 'neutrino' in blob.keys():

        nu = blob['neutrino']

        blob['Neutrino'] = Table({
            'id': np.array(nu)[:, 0].astype(int),
            'pos_x': np.array(nu)[:, 1],
            'pos_y': np.array(nu)[:, 2],
            'pos_z': np.array(nu)[:, 3],
            'dir_x': np.array(nu)[:, 4],
            'dir_y': np.array(nu)[:, 5],
            'dir_z': np.array(nu)[:, 6],
            'energy': np.array(nu)[:, 7],
            'time': np.array(nu)[:, 8],
            'particle_id': np.array(nu)[:, 9].astype(int),
            'is_charm': np.array(nu)[:, 10].astype(int),
            'mother_pid': np.array(nu)[:, 11].astype(int),
            'grandmother_pid': np.array(nu)[:, 11].astype(int),
        },
                                 h5loc='nu')
        blob['NeutrinoMultiplicity'] = Table({
            'total': len(np.array(nu)[:, 6]),
            'nue': len(np.array(nu)[:, 6][np.array(nu)[:, 9] == 12]),
            'anue': len(np.array(nu)[:, 6][np.array(nu)[:, 9] == -12]),
            'numu': len(np.array(nu)[:, 6][np.array(nu)[:, 9] == 14]),
            'anumu': len(np.array(nu)[:, 6][np.array(nu)[:, 9] == -14]),
        },
                                             h5loc='nu_multiplicity')

    if ('track_in' or 'neutrino') in blob.keys():

        blob['Weights'] = Table({
            'w1': blob['weights'][0][0],
            'w2': blob['weights'][0][1],
            'w3': blob['weights'][0][2],
        },
                                h5loc='weights')

    if 'track_primary' in blob.keys():

        primary = blob['track_primary']

        blob['Primary'] = Table({
            'id': np.array(primary)[:, 0].astype(int),
            'pos_x': np.array(primary)[:, 1],
            'pos_y': np.array(primary)[:, 2],
            'pos_z': np.array(primary)[:, 3],
            'dir_x': np.array(primary)[:, 4],
            'dir_y': np.array(primary)[:, 5],
            'dir_z': np.array(primary)[:, 6],
            'energy': np.array(primary)[:, 7],
            'time': np.array(primary)[:, 8],
            'particle_id': np.array(primary)[:, 9].astype(int)
        },
                                h5loc='primary')

    return blob


EVT_PARSERS = {
    'km3sim': Parser(KM3SIM_TAGS),
    'gseagen': Parser(GSEAGEN_TAGS),
    'km3': Parser(KM3_TAGS),
    'propa': parse_propa,
    'corant': parse_corant,
}
