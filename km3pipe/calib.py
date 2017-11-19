# coding=utf-8
# Filename: calib.py
# pylint: disable=locally-disabled
"""
Calibration.

"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd

from .core import Module
from .hardware import Detector
from .dataclasses import (CRawHitSeries, HitSeries, RawHitSeries,
                          CMcHitSeries, McHitSeries)
from .logger import logging

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Thomas Heid"]
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = logging.getLogger(__name__)  # pylint: disable=C0103
# log.setLevel(logging.DEBUG)

try:
    import numba as nb
except ImportError:
    log.debug("No Numba support")
    HAVE_NUMBA = False
else:
    log.debug("Running with Numba support")
    HAVE_NUMBA = True


class Calibration(Module):
    """A very simple, preliminary Module which gives access to the calibration.

    Parameters
    ----------
    apply: bool, optional [default=False]
        Apply the calibration to the hits (add position/direction/t0)?
    filename: str, optional [default=None]
        DetX file with detector description.
    det_id: int, optional
        .detx ID of detector (when retrieving from database).
    t0set: optional
        t0set (when retrieving from database).
    calibration: optional
        calibration (when retrieving from database).
    """
    def configure(self):
        self._should_apply = self.get('apply') or False
        self.filename = self.get('filename') or None
        self.det_id = self.get('det_id') or None
        self.t0set = self.get('t0set') or None
        self.calibration = self.get('calibration') or None
        self.detector = self.get('detector') or None
        self._pos_dom_channel = None
        self._dir_dom_channel = None
        self._t0_dom_channel = None
        self._pos_pmt_id = None
        self._dir_pmt_id = None
        self._t0_pmt_id = None
        self._lookup_tables = None  # for Numba

        if self.filename or self.det_id:
            if self.filename is not None:
                self.detector = Detector(filename=self.filename)
            if self.det_id:
                self.detector = Detector(det_id=self.det_id,
                                         t0set=self.t0set,
                                         calibration=self.calibration)

        if self.detector is not None:
            log.debug("Creating lookup tables")
            self._create_dom_channel_lookup()
            self._create_pmt_id_lookup()
        else:
            log.critical("No detector information loaded.")

    def process(self, blob, key='Hits'):
        if self._should_apply:
            self.apply(blob[key])
        return blob

    def get_detector(self):
        """Return the detector"""
        return self.detector

    def apply_t0(self, hits):
        """Apply only t0s"""
        if HAVE_NUMBA:
            apply_t0_nb(hits.time, hits.dom_id, hits.channel_id,
                        self._lookup_tables)
        else:
            n = len(hits)
            cal = np.empty(n)
            lookup = self._calib_by_dom_and_channel
            for i in range(n):
                calib = lookup[hits._arr['dom_id'][i]][hits._arr['channel_id'][i]]
                cal[i] = calib[6]
            hits.time += cal

    def apply(self, hits):
        """Add x, y, z, t0 (and du, floor if DataFrame) columns to hit.

        When applying to ``RawHitSeries`` or ``McHitSeries``, a ``HitSeries``
        will be returned with the calibration information added.

        """
        if isinstance(hits, RawHitSeries):
            return self._apply_to_rawhitseries(hits)
        elif isinstance(hits, (HitSeries, list)):
            self._apply_to_hitseries(hits)
        elif isinstance(hits, pd.DataFrame):
            self._apply_to_table(hits)
        elif isinstance(hits, McHitSeries):
            return self._apply_to_mchitseries(hits)
        else:
            raise TypeError("Don't know how to apply calibration to '{0}'."
                            .format(hits.__class__.__name__))

    def _apply_to_hitseries(self, hits):
        """Add x, y, z and t0 offset to hit series"""
        for idx, hit in enumerate(hits):
            try:
                pmt = self.detector.get_pmt(hit.dom_id, hit.channel_id)
            except (KeyError, AttributeError):
                pmt = self.detector.pmt_with_id(hit.pmt_id)
            hits.pos_x[idx] = pmt.pos[0]
            hits.pos_y[idx] = pmt.pos[1]
            hits.pos_z[idx] = pmt.pos[2]
            hits.dir_x[idx] = pmt.dir[0]
            hits.dir_y[idx] = pmt.dir[1]
            hits.dir_z[idx] = pmt.dir[2]
            hits._arr['t0'][idx] = pmt.t0
            hits._arr['time'][idx] += pmt.t0
            # hit.a = hit.tot

    def _apply_to_rawhitseries(self, hits):
        """Create a HitSeries from RawHitSeries and add pos, dir and t0.

        Note that existing arrays like tot, dom_id, channel_id etc. will be
        copied by reference for better performance.

        """
        n = len(hits)
        cal = np.empty((n, 9))
        lookup = self._calib_by_dom_and_channel
        for i in range(n):
            calib = lookup[hits._arr['dom_id'][i]][hits._arr['channel_id'][i]]
            cal[i] = calib
        h = np.empty(n, CRawHitSeries.dtype)
        h['channel_id'] = hits.channel_id
        h['dir_x'] = cal[:, 3]
        h['dir_y'] = cal[:, 4]
        h['dir_z'] = cal[:, 5]
        h['dom_id'] = hits.dom_id
        h['du'] = cal[:, 7]
        h['floor'] = cal[:, 8]
        h['pos_x'] = cal[:, 0]
        h['pos_y'] = cal[:, 1]
        h['pos_z'] = cal[:, 2]
        h['t0'] = cal[:, 6]
        h['time'] = hits.time + cal[:, 6]
        h['tot'] = hits.tot
        h['triggered'] = hits.triggered
        h['event_id'] = hits._arr['event_id']
        return CRawHitSeries(h, hits.event_id)

    def _apply_to_mchitseries(self, hits):
        """Create a HitSeries from McHitSeries and add pos, dir and t0.

        Note that existing arrays like a, origin, pmt_id will be copied by
        reference for better performance.

        The attributes ``a`` and ``origin`` are not implemented yet.

        """
        n = len(hits)
        cal = np.empty((n, 9))
        for i in range(n):
            lookup = self._calib_by_pmt_id
            calib = lookup[hits._arr['pmt_id'][i]]
        h = np.empty(n, CMcHitSeries.dtype)
        h['channel_id'] = np.zeros(n, dtype=int)
        h['dir_x'] = cal[:, 3]
        h['dir_y'] = cal[:, 4]
        h['dir_z'] = cal[:, 5]
        h['du'] = cal[:, 7]
        h['floor'] = cal[:, 8]
        h['pmt_id'] = hits._arr['pmt_id']
        h['pos_x'] = cal[:, 0]
        h['pos_y'] = cal[:, 1]
        h['pos_z'] = cal[:, 2]
        h['t0'] = cal[:, 6]
        h['time'] = hits.time + cal[:, 6]
        h['tot'] = np.zeros(n, dtype=int)
        h['triggered'] = np.zeros(n, dtype=bool)
        h['event_id'] = hits._arr['event_id']
        return CMcHitSeries(h, hits.event_id)

    def _apply_to_table(self, table):
        """Add x, y, z and du, floor columns to hit table"""
        def get_pmt(hit):
            return self.detector.get_pmt(hit['dom_id'], hit['channel_id'])

        table['pos_x'] = table.apply(lambda h: get_pmt(h).pos.x, axis=1)
        table['pos_y'] = table.apply(lambda h: get_pmt(h).pos.y, axis=1)
        table['pos_z'] = table.apply(lambda h: get_pmt(h).pos.z, axis=1)
        table['dir_x'] = table.apply(lambda h: get_pmt(h).dir.x, axis=1)
        table['dir_y'] = table.apply(lambda h: get_pmt(h).dir.y, axis=1)
        table['dir_z'] = table.apply(lambda h: get_pmt(h).dir.z, axis=1)
        table['time'] += table.apply(lambda h: get_pmt(h).t0, axis=1)
        table['t0'] = table.apply(lambda h: get_pmt(h).t0, axis=1)
        table['du'] = table.apply(lambda h: get_pmt(h).omkey[0], axis=1)
        table['floor'] = table.apply(lambda h: get_pmt(h).omkey[1], axis=1)

    def _create_dom_channel_lookup(self):
        data = {}
        for dom_id, pmts in self.detector._pmts_by_dom_id.items():
            for pmt in pmts:
                if dom_id not in data:
                    data[dom_id] = np.zeros((31, 9))
                data[dom_id][pmt.channel_id] = [pmt.pos[0],
                                                pmt.pos[1],
                                                pmt.pos[2],
                                                pmt.dir[0],
                                                pmt.dir[1],
                                                pmt.dir[2],
                                                pmt.t0,
                                                pmt.omkey[0],
                                                pmt.omkey[1]]
        self._calib_by_dom_and_channel = data
        if HAVE_NUMBA:
            self._lookup_tables = [(dom, cal) for dom, cal in data.items()]

    def _create_pmt_id_lookup(self):
        data = {}
        for pmt_id, pmt in self.detector._pmts_by_id.items():
            data[pmt_id] = np.array((pmt.pos[0],
                                     pmt.pos[1],
                                     pmt.pos[2],
                                     pmt.dir[0],
                                     pmt.dir[1],
                                     pmt.dir[2],
                                     pmt.t0,
                                     pmt.omkey[0],
                                     pmt.omkey[1],
                                     ))
        self._calib_by_pmt_id = data

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Calibration: det_id({0})".format(self.det_id)


if HAVE_NUMBA:
    log.info("Initialising Numba JIT functions")
    @nb.jit
    def apply_t0_nb(times, dom_ids, channel_ids, lookup_tables):
        """Apply t0s using a lookup table of tuples (dom_id, calib)"""
        dom_id = 0
        lookup = np.empty((31, 9))
        for i in range(len(times)):
            cur_dom_id = dom_ids[i]
            if cur_dom_id != dom_id:
                dom_id = cur_dom_id
                for (d, m) in lookup_tables:
                    if d == dom_id:
                        np.copyto(lookup, m)
            t0 = lookup[channel_ids[i]][6]
            times[i] += t0
