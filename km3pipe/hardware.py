# coding=utf-8
# Filename: hardware.py
# pylint: disable=locally-disabled
"""
Classes representing KM3NeT hardware.

"""
from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

import os

from km3pipe.tools import unpack_nfirst, split
from km3pipe.dataclasses import Point, Direction
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103


class Detector(object):
    """The KM3NeT detector"""
    def __init__(self, filename=None):
        self.det_file = None
        self.det_id = None
        self.n_doms = None
        self.n_pmts_per_dom = None
        self.doms = {}
        self.pmts = {}
        self._pmts = {}

        if filename:
            self.init_from_file(filename)

    def init_from_file(self, filename):
        """Create detector from detx file."""
        file_ext = os.path.splitext(filename)[1][1:]
        if not file_ext == 'detx':
            raise NotImplementedError('Only the detx format is supported.')
        self.open_file(filename)
        self.parse_header()
        self.parse_doms()
        self.det_file.close()

    def open_file(self, filename):
        """Create the file handler"""
        self.det_file = open(filename, 'r')

    def parse_header(self):
        """Extract information from the header of the detector file"""
        self.det_file.seek(0, 0)
        first_line = self.det_file.readline()
        self.det_id, self.n_doms = split(first_line, int)

    # pylint: disable=C0103
    def parse_doms(self):
        """Extract dom information from detector file"""
        self.det_file.seek(0, 0)
        self.det_file.readline()
        lines = self.det_file.readlines()
        try:
            while True:
                line = lines.pop(0)
                if not line:
                    continue
                try:
                    dom_id, line_id, floor_id, n_pmts = split(line, int)
                except ValueError:
                    continue
                self.doms[dom_id] = (line_id, floor_id, n_pmts)
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
                    pmt_entry = tuple([pmt_id] + pmt_pos + pmt_dir + [t0])
                    self.pmts[(line_id, floor_id, i)] = pmt_entry
                    self._pmts[pmt_id] = PMT(pmt_id, pmt_pos, pmt_dir, t0)
        except IndexError:
            pass

    @property
    def dom_positions(self):
        """The positions of the DOMs, taken from the PMT with the lowest ID."""
        return [pmt.pos for pmt in self._pmts.values()
                if pmt.id % self.n_pmts_per_dom == 1]

    def pmt_with_id(self, pmt_id):
        return self._pmts[pmt_id]

    def pmtid2omkey(self, pmt_id, first_pmt_id=1, oms_per_line=18, pmts_per_om=31):
        """Convert (consecutive) raw PMT IDs to Multi-OMKeys."""
        pmts_per_line = oms_per_line * pmts_per_om
        line = ((pmt_id - first_pmt_id) // pmts_per_line) + 1
        om = oms_per_line - (pmt_id - first_pmt_id) % pmts_per_line // pmts_per_om
        pmt = (pmt_id - first_pmt_id) % pmts_per_om
        return int(line), int(om), int(pmt)


class PMT(object):
    def __init__(self, id, pos, dir, t0):
        self.id = id
        self.pos = Point(pos)
        self.dir = Direction(dir)
        self.t0 = t0

    def __str__(self):
        return "PMT id:{0} pos: {1} dir: dir{2} t0: {3}"\
               .format(self.id, self.pos, self.dir, self.t0)