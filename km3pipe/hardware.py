# Filename: hardware.py
# pylint: disable=locally-disabled
"""
Classes representing KM3NeT hardware.

"""

from collections import OrderedDict, defaultdict
from io import StringIO

import numpy as np

from .dataclasses import Table
from .tools import unpack_nfirst, split
from .math import com  # , ignored
from .db import DBManager

from .logger import get_logger, get_printer


log = get_logger(__name__)  # pylint: disable=C0103

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class Detector(object):
    """A KM3NeT detector.

    Parameters
    ----------
    filename: str, optional
        Name of .detx file with detector definition.
    det_id: int, optional
        .detx ID of detector (when retrieving from database).
    t0set: optional
        t0set (when retrieving from database).
    calibration: optional
        calibration (when retrieving from database).
    """

    def __init__(self, filename=None,
                 det_id=None,
                 t0set=None,
                 calibration=None):
        self._det_file = None
        self.det_id = None
        self.n_doms = None
        self.dus = []
        self.n_pmts_per_dom = None
        self.doms = OrderedDict()
        self.pmts = []
        self.version = None
        self.valid_from = None
        self.valid_until = None
        self.utm_info = None
        self._dom_ids = []
        self._dom_positions = OrderedDict()
        self._pmt_index_by_omkey = OrderedDict()
        self._pmt_index_by_pmt_id = OrderedDict()
        self._pmt_angles = []
        self._xy_pos = []
        self._current_du = None

        self.print = get_printer(self.__class__.__name__)

        if filename:
            self._init_from_file(filename)

        if det_id is not None:
            self.print("Retrieving DETX with detector ID {0} "
                       "from the database..."
                       .format(det_id))
            db = DBManager()
            detx = db.detx(det_id, t0set=t0set, calibration=calibration)
            self._det_file = StringIO(detx)
            self._parse_header()
            self._parse_doms()
            if self.n_doms < 1:
                log.error("No data found for detector ID %s." % det_id)

    def _init_from_file(self, filename):
        """Create detector from detx file."""
        if not filename.endswith("detx"):
            raise NotImplementedError('Only the detx format is supported.')
        self._open_file(filename)
        self._parse_header()
        self._parse_doms()
        self._det_file.close()

    def _open_file(self, filename):
        """Create the file handler"""
        self._det_file = open(filename, 'r')

    def _parse_header(self):
        """Extract information from the header of the detector file"""
        self.print("Parsing the DETX header")
        self._det_file.seek(0, 0)
        first_line = self._det_file.readline()
        try:
            self.det_id, self.n_doms = split(first_line, int)
            self.version = 'v1'
        except ValueError:
            det_id, self.version = first_line.split()
            self.det_id = int(det_id)
            validity = self._det_file.readline().strip()
            self.valid_from, self.valid_until = split(validity, float)
            raw_utm_info = self._det_file.readline().strip().split(' ')
            try:
                self.utm_info = UTMInfo(*raw_utm_info[1:])
            except TypeError:
                log.warning("Missing UTM information.")
            n_doms = self._det_file.readline()
            self.n_doms = int(n_doms)

    # pylint: disable=C0103
    def _parse_doms(self):
        """Extract dom information from detector file"""
        self.print("Reading PMT information...")
        self._det_file.seek(0, 0)
        self._det_file.readline()
        pmts = defaultdict(list)
        pmt_index = 0
        while True:
            line = self._det_file.readline()

            if line == '':
                self.print("Done.")
                break

            try:
                dom_id, du, floor, n_pmts = split(line, int)
            except ValueError:
                continue

            if du != self._current_du:
                log.debug("Next DU, resetting floor to 1.")
                self._current_du = du
                self.dus.append(du)
                self._current_floor = 1
                if du == 1 and floor == -1:
                    log.warning("Floor ID is -1 (Jpp conversion bug), "
                                "using our own floor ID!")
            else:
                self._current_floor += 1

            if floor == -1:
                log.debug("Setting floor ID to our own ID")
                floor = self._current_floor

            self.doms[dom_id] = (du, floor, n_pmts)

            if self.n_pmts_per_dom is None:
                self.n_pmts_per_dom = n_pmts

            if self.n_pmts_per_dom != n_pmts:
                log.warning("DOMs with different number of PMTs are "
                            "detected, this can cause some unexpected "
                            "behaviour.")

            for i in range(n_pmts):
                raw_pmt_info = self._det_file.readline()
                pmt_info = raw_pmt_info.split()
                pmt_id, x, y, z, rest = unpack_nfirst(pmt_info, 4)
                dx, dy, dz, t0, rest = unpack_nfirst(rest, 4)
                if rest:
                    log.warning("Unexpected PMT values: {0}".format(rest))
                pmt_id = int(pmt_id)
                omkey = (du, floor, i)
                pmts['pmt_id'].append(int(pmt_id))
                pmts['pos_x'].append(float(x))
                pmts['pos_y'].append(float(y))
                pmts['pos_z'].append(float(z))
                pmts['dir_x'].append(float(dx))
                pmts['dir_y'].append(float(dy))
                pmts['dir_z'].append(float(dz))
                pmts['t0'].append(float(t0))
                pmts['du'].append(int(du))
                pmts['floor'].append(int(floor))
                pmts['channel_id'].append(int(i))
                pmts['dom_id'].append(int(dom_id))
                self._pmt_index_by_omkey[omkey] = pmt_index
                self._pmt_index_by_pmt_id[pmt_id] = pmt_index
                pmt_index += 1

        self.pmts = Table(pmts, name='PMT')

    @property
    def dom_ids(self):
        if not self._dom_ids:
            self._dom_ids = list(self.doms.keys())
        return self._dom_ids

    @property
    def dom_positions(self):
        """The positions of the DOMs, calculated as COM from PMTs."""
        if not self._dom_positions:
            for dom_id in self.dom_ids:
                mask = self.pmts.dom_id == dom_id
                pmt_positions = self.pmts[mask].pos
                self._dom_positions[dom_id] = com(pmt_positions)
        return self._dom_positions

    @property
    def xy_positions(self):
        """XY positions of all doms."""
        if not self._xy_pos:
            dom_pos = self.dom_positions
            xy = np.unique([(pos[0], pos[1]) for pos in dom_pos.values()],
                           axis=0)
            self._xy_pos = xy
        return self._xy_pos

    def translate_detector(self, vector):
        vector = np.array(vector, dtype=float)
        for pmt in self.pmts:
            pmt.pos = pmt.pos + vector

    @property
    def pmt_angles(self):
        """A list of PMT directions sorted by PMT channel"""
        if not self._pmt_angles:
            pmts = self.pmts[:self.n_pmts_per_dom]
            self._pmt_angles = [pmt.dir for pmt in pmts]
        return self._pmt_angles

    @property
    def ascii(self):
        """The ascii representation of the detector"""
        if self.version == 'v1':
            header = "{det.det_id} {det.n_doms}".format(det=self)
        else:
            header = "{det.det_id} {det.version}".format(det=self)
            header += "\n{0} {1}".format(self.valid_from, self.valid_until)
            header += "\n" + str(self.utm_info) + "\n"
            header += str(self.n_doms)

        doms = ""
        for dom_id, (line, floor, n_pmts) in self.doms.items():
            doms += "{0} {1} {2} {3}\n".format(dom_id, line, floor, n_pmts)
            for channel_id in range(n_pmts):
                pmt_idx = self._pmt_index_by_omkey[(line, floor, channel_id)]
                pmt = self.pmts[pmt_idx]
                doms += " {0} {1} {2} {3} {4} {5} {6} {7}\n".format(
                        pmt.pmt_id, pmt.pos_x, pmt.pos_y, pmt.pos_z,
                        pmt.dir_x, pmt.dir_y, pmt.dir_z,
                        pmt.t0
                )
        return header + "\n" + doms

    def write(self, filename):
        """Save detx file."""
        with open(filename, 'w') as f:
            f.write(self.ascii)
        self.print("Detector file saved as '{0}'".format(filename))

    def pmt_with_id(self, pmt_id):
        """Get PMT with global pmt_id"""
        try:
            return self.pmts[self._pmt_index_by_pmt_id[pmt_id]]
        except KeyError:
            raise KeyError("No PMT found for ID: {0}".format(pmt_id))

    def get_pmt(self, dom_id, channel_id):
        """Return PMT with DOM ID and DAQ channel ID"""
        du, floor, _ = self.doms[dom_id]
        pmt = self.pmts[self._pmt_index_by_omkey[(du, floor, channel_id)]]
        return pmt

    def pmtid2omkey(self, pmt_id):
        return self._pmts_by_id[int(pmt_id)].omkey

    def domid2floor(self, dom_id):
        _, floor, _ = self.doms[dom_id]
        return floor

    @property
    def n_dus(self):
        return len(self.dus)

    def __str__(self):
        return "Detector id: '{0}', n_doms: {1}, dus: {2}".format(
            self.det_id, self.n_doms, self.dus)

    def __repr__(self):
        return self.__str__()


class UTMInfo(object):
    """The UTM information"""
    def __init__(self, ellipsoid, grid, easting, northing, z):
        self.ellipsoid = ellipsoid
        self.grid = grid
        self.easting = float(easting)
        self.northing = float(northing)
        self.z = float(z)

    def __str__(self):
        return "UTM {} {} {} {} {}"  \
               .format(self.ellipsoid, self.grid,
                       self.easting, self.northing, self.z)

    def __repr__(self):
        return "UTMInfo: {}".format(self)


class PMT(object):
    """Represents a photomultiplier.

    Parameters
    ----------
    id: int
    pos: 3-float-tuple (x, y, z)
    dir: 3-float-tuple (x, y, z)
    t0: int
    channel_id: int
    omkey: int
    """

    def __init__(self, id, pos, dir, t0, channel_id, omkey):
        self.id = id
        self.pos = pos
        self.dir = dir
        self.t0 = t0
        self.channel_id = channel_id
        self.omkey = omkey

    def __str__(self):
        return "PMT id:{0}, pos: {1}, dir: dir{2}, t0: {3}, DAQ channel: {4}"\
               .format(self.id, self.pos, self.dir, self.t0, self.channel_id)


# PMT DAQ channel IDs ordered from top to bottom
ORDERED_PMT_IDS = [28, 23, 22, 21, 27, 29, 20, 30, 26, 25, 19, 24, 13, 7, 1,
                   14, 18, 12, 6, 2, 11, 8, 0, 15, 4, 3, 5, 17, 10, 9, 16]
