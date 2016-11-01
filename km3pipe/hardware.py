# coding=utf-8
# Filename: hardware.py
# pylint: disable=locally-disabled
"""
Classes representing KM3NeT hardware.

"""
from __future__ import division, absolute_import, print_function

from collections import OrderedDict, defaultdict
import os
import sys

import numpy as np

from km3pipe.tools import unpack_nfirst, split, com  # , ignored
from km3pipe.dataclasses import Point, Direction
from km3pipe.db import DBManager

from km3pipe.logger import logging

if sys.version_info[0] > 2:
    from io import StringIO
else:
    from StringIO import StringIO


log = logging.getLogger(__name__)  # pylint: disable=C0103

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
        self.lines = set()
        self.n_pmts_per_dom = None
        self.doms = OrderedDict()
        self.pmts = []
        self.version = None
        self.valid_from = None
        self.valid_until = None
        self.utm_info = None
        self._dom_ids = []
        self._dom_positions = OrderedDict()
        self._pmts_by_omkey = OrderedDict()
        self._pmts_by_id = OrderedDict()
        self._pmts_by_dom_id = defaultdict(list)
        self._pmt_angles = []

        if filename:
            self._init_from_file(filename)

        if det_id is not None:
            print("Retrieving DETX with detector ID {0} from the database..."
                  .format(det_id))
            db = DBManager()
            detx = db.detx(det_id, t0set=t0set, calibration=calibration)
            self._det_file = StringIO(detx)
            self._parse_header()
            self._parse_doms()
            if self.n_doms < 1:
                log.error("No data found for detector ID {0}.".format(det_id))

    def _init_from_file(self, filename):
        """Create detector from detx file."""
        file_ext = os.path.splitext(filename)[1][1:]
        if not file_ext == 'detx':
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
        self._det_file.seek(0, 0)
        first_line = self._det_file.readline()
        try:
            self.det_id, self.n_doms = split(first_line, int)
            self.version = 'v1'
        except ValueError:
            det_id, self.version = first_line.split()
            self.det_id = int(det_id)
            validity = self._det_file.readline()
            self.valid_from, self.valid_until = split(validity, float)
            self.utm_info = self._det_file.readline()
            n_doms = self._det_file.readline()
            self.n_doms = int(n_doms)

    # pylint: disable=C0103
    def _parse_doms(self):
        """Extract dom information from detector file"""
        self._det_file.seek(0, 0)
        self._det_file.readline()
        lines = self._det_file.readlines()
        try:
            while True:
                line = lines.pop(0)
                if not line:
                    continue
                try:
                    dom_id, line_id, floor_id, n_pmts = split(line, int)
                except ValueError:
                    continue
                self.lines.add(line_id)
                self.n_pmts_per_dom = n_pmts
                for i in range(n_pmts):
                    raw_pmt_info = lines.pop(0)
                    pmt_info = raw_pmt_info.split()
                    pmt_id, x, y, z, rest = unpack_nfirst(pmt_info, 4)
                    dx, dy, dz, t0, rest = unpack_nfirst(rest, 4)
                    if rest:
                        log.warn("Unexpected PMT values: '{0}'".format(rest))
                    pmt_id = int(pmt_id)
                    pmt_pos = [float(n) for n in (x, y, z)]
                    pmt_dir = [float(n) for n in (dx, dy, dz)]
                    t0 = float(t0)
                    if floor_id < 0:
                        _, new_floor_id, _ = self._pmtid2omkey_old(pmt_id)
                        log.error("Floor ID is negative for PMT {0}.\n"
                                  "Guessing correct id: {1}"
                                  .format(pmt_id, new_floor_id))
                        floor_id = new_floor_id
                    # TODO: following line is here bc of the bad MC floor IDs
                    #      put it outside the for loop in future
                    self.doms[dom_id] = (line_id, floor_id, n_pmts)
                    omkey = (line_id, floor_id, i)
                    pmt = PMT(pmt_id, pmt_pos, pmt_dir, t0, i, omkey)
                    self.pmts.append(pmt)
                    self._pmts_by_omkey[(line_id, floor_id, i)] = pmt
                    self._pmts_by_id[pmt_id] = pmt
                    self._pmts_by_dom_id[dom_id].append(pmt)
        except IndexError:
            pass

    @property
    def dom_ids(self):
        if not self._dom_ids:
            self._dom_ids = self.doms.keys()
        return self._dom_ids

    @property
    def dom_positions(self):
        """The positions of the DOMs, calculated as COM from PMTs."""
        if not self._dom_positions:
            for dom_id in self.dom_ids:
                pmt_positions = [p.pos for p in self._pmts_by_dom_id[dom_id]]
                self._dom_positions[dom_id] = com(pmt_positions)
        return self._dom_positions

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
            header += "\n" + self.utm_info
            header += str(self.n_doms)

        doms = ""
        for dom_id, (line, floor, n_pmts) in self.doms.iteritems():
            doms += "{0} {1} {2} {3}\n".format(dom_id, line, floor, n_pmts)
            for i in xrange(n_pmts):
                pmt = self._pmts_by_omkey[(line, floor, i)]
                doms += "{0} {1} {2} {3} {4} {5} {6} {7}\n".format(
                        pmt.id, pmt.pos[0], pmt.pos[1], pmt.pos[2],
                        pmt.dir[0], pmt.dir[1], pmt.dir[2],
                        pmt.t0
                        )
        return header + "\n" + doms

    def write(self, filename):
        """Save detx file."""
        with open(filename, 'w') as f:
            f.write(self.ascii)
        print("Detector file saved as '{0}'".format(filename))

    def pmt_with_id(self, pmt_id):
        """Get PMT with global pmt_id"""
        try:
            return self._pmts_by_id[pmt_id]
        except KeyError:
            raise KeyError("No PMT found for ID: {0}".format(pmt_id))

    def get_pmt(self, dom_id, channel_id):
        """Return PMT with DOM ID and DAQ channel ID"""
        line, floor, _ = self.doms[dom_id]
        pmt = self._pmts_by_omkey[(line, floor, channel_id)]
        return pmt

    def pmtid2omkey(self, pmt_id):
        return self._pmts_by_id[int(pmt_id)].omkey

    def _pmtid2omkey_old(self, pmt_id,
                         first_pmt_id=1, oms_per_line=18, pmts_per_om=31):
        """Convert (consecutive) raw PMT IDs to Multi-OMKeys."""
        pmts_line = oms_per_line * pmts_per_om
        line = ((pmt_id - first_pmt_id) // pmts_line) + 1
        om = oms_per_line - (pmt_id - first_pmt_id) % pmts_line // pmts_per_om
        pmt = (pmt_id - first_pmt_id) % pmts_per_om
        return int(line), int(om), int(pmt)

    def domid2floor(self, dom_id):
        _, floor, _ = self.doms[dom_id]
        return floor

    @property
    def n_lines(self):
        return len(self.lines)

    def __str__(self):
        return "Detector id: '{0}', n_doms: {1}, n_lines: {2}".format(
            self.det_id, self.n_doms, self.n_lines)

    def __repr__(self):
        return self.__str__()


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
        self.pos = Point(pos)
        self.dir = Direction(dir)
        self.t0 = t0
        self.channel_id = channel_id
        self.omkey = omkey

    def __str__(self):
        return "PMT id:{0}, pos: {1}, dir: dir{2}, t0: {3}, DAQ channel: {4}"\
               .format(self.id, self.pos, self.dir, self.t0, self.channel_id)
