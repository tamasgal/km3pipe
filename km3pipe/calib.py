# Filename: calib.py
# pylint: disable=locally-disabled
"""
Calibration.

"""
from __future__ import absolute_import, print_function, division

import numpy as np

from .core import Module
from .hardware import Detector
from .dataclasses import Table
from .tools import istype

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Thomas Heid"]
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

try:
    import numba as nb
except (ImportError, OSError):
    HAVE_NUMBA = False
else:
    HAVE_NUMBA = True


class Calibration(Module):
    """A very simple, preliminary Module which gives access to the calibration.

    Parameters
    ----------
    apply: bool, optional [default=True]
        Apply the calibration to the hits (add position/direction/t0)?
    filename: str, optional [default=None]
        DetX file with detector description.
    det_id: int, optional
        .detx ID of detector (when retrieving from database).
    t0set: optional
        t0set (when retrieving from database).
    calibset: optional
        calibset (when retrieving from database).
    """
    __name__ = 'Calibration'
    name = 'Calibration'

    def configure(self):
        self._should_apply = self.get('apply', default=True)
        self.filename = self.get('filename')
        self.det_id = self.get('det_id')
        self.t0set = self.get('t0set')
        self.calibset = self.get('calibset')
        self.detector = self.get('detector')
        self._pos_dom_channel = None
        self._dir_dom_channel = None
        self._t0_dom_channel = None
        self._pos_pmt_id = None
        self._dir_pmt_id = None
        self._t0_pmt_id = None
        self._lookup_tables = None    # for Numba

        # TODO: deprecation
        if self.get('calibration'):
            self.log.warning(
                "The parameter 'calibration' has been renamed "
                "to 'calibset'. The 'calibration' parameter will be removed "
                "in the next version of KM3Pipe"
            )
            self.calibset = self.get('calibration')

        if self.filename or self.det_id:
            if self.filename is not None:
                self.detector = Detector(filename=self.filename)
            if self.det_id:
                self.detector = Detector(
                    det_id=self.det_id,
                    t0set=self.t0set,
                    calibration=self.calibset
                )

        if self.detector is not None:
            self.log.debug("Creating lookup tables")
            self._create_dom_channel_lookup()
            self._create_pmt_id_lookup()
        else:
            self.log.critical("No detector information loaded.")

    def process(self, blob, key='Hits', outkey='CalibHits'):
        if self._should_apply:
            blob[outkey] = self.apply(blob[key])
        return blob

    def get_detector(self):
        """Return the detector"""
        return self.detector

    def apply_t0(self, hits):
        """Apply only t0s"""
        if HAVE_NUMBA:
            apply_t0_nb(
                hits.time, hits.dom_id, hits.channel_id, self._lookup_tables
            )
        else:
            n = len(hits)
            cal = np.empty(n)
            lookup = self._calib_by_dom_and_channel
            for i in range(n):
                calib = lookup[hits['dom_id'][i]][hits['channel_id'][i]]
                cal[i] = calib[6]
            hits.time += cal
        return hits

    def apply(self, hits):
        """Add x, y, z, t0 (and du, floor if DataFrame) columns to the hits.

        """
        if istype(hits, 'DataFrame'):
            # do we ever see McHits here?
            hits = Table.from_template(hits, 'Hits')
        if hasattr(hits, 'dom_id') and hasattr(hits, 'channel_id'):
            return self._apply_to_hits(hits)
        elif hasattr(hits, 'pmt_id'):
            return self._apply_to_mchits(hits)
        else:
            raise TypeError(
                "Don't know how to apply calibration to '{0}'. "
                "We need at least 'dom_id' and 'channel_id', or "
                "'pmt_id'.".format(hits.name)
            )

    def _apply_to_hits(self, hits):
        """Append the position, direction and t0 columns and add t0 to time"""
        n = len(hits)
        cal = np.empty((n, 9))
        lookup = self._calib_by_dom_and_channel
        for i in range(n):
            calib = lookup[hits['dom_id'][i]][hits['channel_id'][i]]
            cal[i] = calib
        dir_x = cal[:, 3]
        dir_y = cal[:, 4]
        dir_z = cal[:, 5]
        du = cal[:, 7]
        floor = cal[:, 8]
        pos_x = cal[:, 0]
        pos_y = cal[:, 1]
        pos_z = cal[:, 2]

        t0 = cal[:, 6]

        if hits.time.dtype != t0.dtype:
            time = hits.time.astype('f4') + t0.astype('f4')
            hits = hits.drop_columns(['time'])
            hits = hits.append_columns(['time'], [time])
        else:
            hits.time += t0

        return hits.append_columns([
            'dir_x', 'dir_y', 'dir_z', 'du', 'floor', 'pos_x', 'pos_y',
            'pos_z', 't0'
        ],
                                   [
                                       dir_x, dir_y, dir_z, du, floor, pos_x,
                                       pos_y, pos_z, t0
                                   ])

    def _apply_to_mchits(self, hits):
        """Append the position, direction and t0 columns and add t0 to time"""
        n_hits = len(hits)
        cal = np.empty((n_hits, 9))
        for i in range(n_hits):
            lookup = self._calib_by_pmt_id
            cal[i] = lookup[hits['pmt_id'][i]]
        dir_x = cal[:, 3]
        dir_y = cal[:, 4]
        dir_z = cal[:, 5]
        du = cal[:, 7]
        floor = cal[:, 8]
        pos_x = cal[:, 0]
        pos_y = cal[:, 1]
        pos_z = cal[:, 2]
        t0 = cal[:, 6]

        hits.time += t0

        return hits.append_columns([
            'dir_x', 'dir_y', 'dir_z', 'du', 'floor', 'pos_x', 'pos_y',
            'pos_z', 't0'
        ],
                                   [
                                       dir_x, dir_y, dir_z, du, floor, pos_x,
                                       pos_y, pos_z, t0
                                   ])

    def _create_dom_channel_lookup(self):
        data = {}
        for pmt in self.detector.pmts:
            if pmt.dom_id not in data:
                data[pmt.dom_id] = np.zeros((31, 9))
            data[pmt.dom_id][pmt.channel_id] = [
                pmt.pos_x, pmt.pos_y, pmt.pos_z, pmt.dir_x, pmt.dir_y,
                pmt.dir_z, pmt.t0, pmt.du, pmt.floor
            ]
        self._calib_by_dom_and_channel = data
        if HAVE_NUMBA:
            self._lookup_tables = [(dom, cal) for dom, cal in data.items()]

    def _create_pmt_id_lookup(self):
        data = {}
        for pmt in self.detector.pmts:
            data[pmt.pmt_id] = np.array((
                pmt.pos_x,
                pmt.pos_y,
                pmt.pos_z,
                pmt.dir_x,
                pmt.dir_y,
                pmt.dir_z,
                pmt.t0,
                pmt.du,
                pmt.floor,
            ))
        self._calib_by_pmt_id = data

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Calibration: det_id({0})".format(self.det_id)


if HAVE_NUMBA:

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


class CalibrationService(Module):
    """A service which provides calibration routines for hits

    Parameters
    ----------
    filename: str, optional [default=None]
        DetX file with detector description.
    det_id: int, optional
        .detx ID of detector (when retrieving from database).
    t0set: optional
        t0set (when retrieving from database).
    calibset: optional
        calibset (when retrieving from database).
    detector: kp.hardware.Detector, optional
    """
    __name__ = 'Calibration'
    name = 'Calibration'

    def configure(self):
        filename = self.get('filename')
        det_id = self.get('det_id')
        t0set = self.get('t0set')
        calibset = self.get('calibset')
        detector = self.get('detector')

        self._calibration = Calibration(
            filename=filename,
            det_id=det_id,
            t0set=t0set,
            calibset=calibset,
            detector=detector
        )

        self.expose(self.calibrate, "calibrate")
        self.expose(self._calibration.detector, "detector")

    def calibrate(self, hits):
        return self._calibration.apply(hits)
