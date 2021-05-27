# Filename: calib.py
# pylint: disable=locally-disabled
"""
Calibration.

"""
import awkward as ak
import numba as nb
import numpy as np

import km3db
import km3io

from thepipe import Module
from km3pipe.hardware import Detector
from km3pipe.dataclasses import Table
from km3pipe.tools import istype
from km3pipe.logger import get_logger

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Thomas Heid"]
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = get_logger(__name__)


class Calibration(Module):
    """A module which applies time, position and rotation corrections to hits.

    This module also calibrates MC hits, but be aware, t0s are not appended to
    the MC hit times.
    Additionally, the global PMT ID is added to regular hits as ``pmt_id`` and
    in case of MC hits, the ``dom_id`` and ``channel_id`` (DAQ) are set.

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
    key: str, optional [default="Hits"]
        the blob key of the hits
    outkey: str, optional [default="CalibHits"]
        the output blob key of the calibrated hits
    key_mc: str, optional [default="McHits"]
        the blob key of the MC hits (if present)
    outkey_mc: str, optional [default="CalibMcHits"]
        the output blob key of the calibrated MC hits
    """

    __name__ = "Calibration"
    name = "Calibration"

    def configure(self):
        self._should_apply = self.get("apply", default=True)
        self.filename = self.get("filename")
        self.det_id = self.get("det_id")
        self.run = self.get("run")
        self.t0set = self.get("t0set")
        self.calibset = self.get("calibset")
        self.detector = self.get("detector")
        self.key = self.get("key", default="Hits")
        self.outkey = self.get("outkey", default="CalibHits")
        self.key_mc = self.get("key_mc", default="McHits")
        self.outkey_mc = self.get("outkey_mc", default="CalibMcHits")
        self._pos_dom_channel = None
        self._dir_dom_channel = None
        self._t0_dom_channel = None
        self._pos_pmt_id = None
        self._dir_pmt_id = None
        self._t0_pmt_id = None
        self._lookup_tables = None  # for Numba

        if self.det_id and self.run:
            self.cprint(
                "Grabbing the calibration for Det ID {} and run {}".format(
                    self.det_id, self.run
                )
            )
            raw_detx = km3db.tools.detx_for_run(self.det_id, self.run)
            self.detector = Detector(string=raw_detx)
            self._create_dom_channel_lookup()
            self._create_pmt_id_lookup()
            return

        if self.filename or self.det_id:
            if self.filename is not None:
                self.detector = Detector(filename=self.filename)
            if self.det_id:
                self.detector = Detector(
                    det_id=self.det_id, t0set=self.t0set, calibset=self.calibset
                )

        if self.detector is not None:
            self.log.debug("Creating lookup tables")
            self._create_dom_channel_lookup()
            self._create_pmt_id_lookup()
        else:
            self.log.critical("No detector information loaded.")

    def process(self, blob):
        if self._should_apply:
            blob[self.outkey] = self.apply(blob[self.key])
            if self.key_mc in blob:
                blob[self.outkey_mc] = self.apply(blob[self.key_mc])
        return blob

    def get_detector(self):
        """Return the detector"""
        return self.detector

    def apply_t0(self, hits):
        """Apply only t0s"""
        apply_t0_nb(hits.time, hits.dom_id, hits.channel_id, self._lookup_tables)
        return hits

    def apply(self, hits, no_copy=False, correct_slewing=True, slewing_variant=3):
        """Add x, y, z, t0 (and du, floor if DataFrame) columns to the hits."""
        if not no_copy:
            try:
                hits = hits.copy()
            except AttributeError:  # probably a km3io object
                pass

        if isinstance(hits, (ak.Array, ak.Record, km3io.rootio.Branch)):
            hits = Table(
                dict(
                    dom_id=hits.dom_id,
                    channel_id=hits.channel_id,
                    time=hits.t,
                    tot=hits.tot,
                    triggered=hits.trig,
                )
            )

        if istype(hits, "DataFrame"):
            # do we ever see McHits here?
            hits = Table.from_template(hits, "Hits")

        is_mc = None
        if hasattr(hits, "dom_id") and hasattr(hits, "channel_id"):
            try:
                (
                    dir_x,
                    dir_y,
                    dir_z,
                    du,
                    floor,
                    pos_x,
                    pos_y,
                    pos_z,
                    t0,
                    pmt_id,
                ) = _get_calibration_for_hits(hits, self._calib_by_dom_and_channel)
            except KeyError as e:
                self.log.critical("Wrong calibration (DETX) data provided.")
                raise
            is_mc = False
        elif hasattr(hits, "pmt_id"):
            try:
                (
                    dir_x,
                    dir_y,
                    dir_z,
                    du,
                    floor,
                    pos_x,
                    pos_y,
                    pos_z,
                    t0,
                    dom_id,
                    channel_id,
                ) = _get_calibration_for_mchits(hits, self._calib_by_pmt_id)
            except KeyError as e:
                self.log.critical("Wrong calibration (DETX) data provided.")
                raise
            is_mc = True
        else:
            raise TypeError(
                "Don't know how to apply calibration to '{0}'. "
                "We need at least 'dom_id' and 'channel_id', or "
                "'pmt_id'.".format(hits.name)
            )

        if hasattr(hits, "time") and not is_mc:
            if hits.time.dtype != t0.dtype:
                time = hits.time.astype("f4") + t0.astype("f4")
                hits = hits.drop_columns(["time"])
                hits = hits.append_columns(["time"], [time])
            else:
                hits.time += t0

        hits_data = {}
        for colname in hits.dtype.names:
            hits_data[colname] = hits[colname]
        calib = {
            "dir_x": dir_x,
            "dir_y": dir_y,
            "dir_z": dir_z,
            "du": du.astype(np.uint8),
            "floor": floor.astype(np.uint8),
            "pos_x": pos_x,
            "pos_y": pos_y,
            "pos_z": pos_z,
            "t0": t0,
        }

        if is_mc:
            calib["dom_id"] = dom_id.astype(np.int32)
            calib["channel_id"] = channel_id.astype(np.int32)
        else:
            calib["pmt_id"] = pmt_id.astype(np.int32)

        hits_data.update(calib)

        if correct_slewing and not is_mc:
            hits_data["time"] -= slew(hits_data["tot"], variant=slewing_variant)
        return Table(
            hits_data, h5loc=hits.h5loc, split_h5=hits.split_h5, name=hits.name
        )

    def _create_dom_channel_lookup(self):
        data = nb.typed.Dict.empty(
            key_type=nb.types.i8, value_type=nb.types.float64[:, :]
        )
        for pmt in self.detector.pmts:
            if pmt.dom_id not in data:
                data[pmt.dom_id] = np.zeros((31, 10))
            data[pmt.dom_id][pmt.channel_id] = np.asarray(
                [
                    pmt.pos_x,
                    pmt.pos_y,
                    pmt.pos_z,
                    pmt.dir_x,
                    pmt.dir_y,
                    pmt.dir_z,
                    pmt.t0,
                    pmt.du,
                    pmt.floor,
                    pmt.pmt_id,
                ],
                dtype=np.float64,
            )
        self._calib_by_dom_and_channel = data
        self._lookup_tables = [(dom, cal) for dom, cal in data.items()]

    def _create_pmt_id_lookup(self):
        data = nb.typed.Dict.empty(key_type=nb.types.i8, value_type=nb.types.float64[:])
        for pmt in self.detector.pmts:
            data[pmt.pmt_id] = np.asarray(
                [
                    pmt.pos_x,
                    pmt.pos_y,
                    pmt.pos_z,
                    pmt.dir_x,
                    pmt.dir_y,
                    pmt.dir_z,
                    pmt.t0,
                    pmt.du,
                    pmt.floor,
                    pmt.dom_id,
                    pmt.channel_id,
                ],
                dtype=np.float64,
            )
        self._calib_by_pmt_id = data

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Calibration: det_id({0})".format(self.det_id)


@nb.njit
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


@nb.jit
def _get_calibration_for_hits(hits, lookup):
    """Append the position, direction and t0 columns and add t0 to time"""
    n = len(hits)
    cal = np.empty((n, 10))
    for i in range(n):
        calib = lookup[hits["dom_id"][i]][hits["channel_id"][i]]
        cal[i] = calib
    dir_x = cal[:, 3]
    dir_y = cal[:, 4]
    dir_z = cal[:, 5]
    du = cal[:, 7]
    floor = cal[:, 8]
    pos_x = cal[:, 0]
    pos_y = cal[:, 1]
    pos_z = cal[:, 2]
    pmt_id = cal[:, 9]

    t0 = cal[:, 6]

    return [dir_x, dir_y, dir_z, du, floor, pos_x, pos_y, pos_z, t0, pmt_id]


@nb.jit
def _get_calibration_for_mchits(hits, lookup):
    """Append the position, direction and t0 columns and add t0 to time"""
    n_hits = len(hits)
    cal = np.empty((n_hits, 11))
    for i in range(n_hits):
        cal[i] = lookup[hits["pmt_id"][i]]
    dir_x = cal[:, 3]
    dir_y = cal[:, 4]
    dir_z = cal[:, 5]
    du = cal[:, 7]
    floor = cal[:, 8]
    pos_x = cal[:, 0]
    pos_y = cal[:, 1]
    pos_z = cal[:, 2]
    t0 = cal[:, 6]
    dom_id = cal[:, 9]
    channel_id = cal[:, 10]

    return [dir_x, dir_y, dir_z, du, floor, pos_x, pos_y, pos_z, t0, dom_id, channel_id]


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

    __name__ = "Calibration"
    name = "Calibration"

    def configure(self):
        self.filename = self.get("filename")
        self.det_id = self.get("det_id")
        self.t0set = self.get("t0set")
        self.calibset = self.get("calibset")

        self._detector = self.get("detector")

        if self._detector is not None:
            self._calibration = Calibration(detector=self._detector)

        self._calibration = None

        self.expose(self.calibrate, "calibrate")
        self.expose(self.get_detector, "get_detector")
        self.expose(self.get_calibration, "get_calibration")
        self.expose(self.load_calibration, "load_calibration")
        self.expose(self.correct_slewing, "correct_slewing")

    def load_calibration(self, filename=None, det_id=None, t0set=None, calibset=None):
        """Load another calibration"""
        self.filename = filename
        self.det_id = det_id
        self.t0set = t0set
        self.calibset = calibset
        self._detector = None
        self._calibration = None

    def calibrate(self, hits, correct_slewing=True):
        return self.calibration.apply(hits, correct_slewing=correct_slewing)

    @property
    def detector(self):
        if self._detector is None:
            self._detector = self.calibration.detector
        return self._detector

    def get_detector(self):
        """Extra getter to be as lazy as possible (expose triggers otherwise"""
        return self.detector

    @property
    def calibration(self):
        if self._calibration is None:
            self._calibration = Calibration(
                filename=self.filename,
                det_id=self.det_id,
                t0set=self.t0set,
                calibset=self.calibset,
            )
        return self._calibration

    def get_calibration(self):
        """Extra getter to be as lazy as possible (expose triggers otherwise"""
        return self.calibration

    def correct_slewing(self, hits):
        """Apply time slewing correction to the hit times"""
        hits.time -= slew(hits.tot)


def slew(tot, variant=3):
    """Calculate the time slewing of a PMT response for a given ToT


    Parameters
    ----------
    tot: int or np.array(int)
        Time over threshold value of a hit
    variant: int, optional
        The variant of the slew calculation.
        1: The first parametrisation approach
        2: Jannik's improvement of the parametrisation
        3: The latest lookup table approach based on lab measurements.

    Returns
    -------
    time: int
        Time slewing, which has to be subtracted from the original hit time.
    """
    if variant == 1:
        return _slew_parametrised(7.70824, 0.00879447, -0.0621101, -1.90226, tot)
    if variant == 2:
        return _slew_parametrised(
            13.6488662517, -0.128744123166, -0.0174837749244, -4.47119633965, tot
        )
    if variant == 3:
        if isinstance(tot, (int, np.int8, np.int16, np.int32, np.int64)):
            return _SLEWS[tot]
        return _slew_tabulated(np.array(_SLEWS), tot)

    raise ValueError("Unknown slew calculation variant '{}'".format(variant))


@nb.jit
def _slew_parametrised(p0, p1, p2, p3, tot):
    return p0 * np.exp(p1 * np.sqrt(tot) + p2 * tot) + p3


@nb.jit
def _slew_tabulated(tab, tots):
    n = len(tots)
    out = np.empty(n)
    for i in range(n):
        out[i] = tab[tots[i]]
    return out


_SLEWS = np.array(
    [
        8.01,
        7.52,
        7.05,
        6.59,
        6.15,
        5.74,
        5.33,
        4.95,
        4.58,
        4.22,
        3.89,
        3.56,
        3.25,
        2.95,
        2.66,
        2.39,
        2.12,
        1.87,
        1.63,
        1.40,
        1.19,
        0.98,
        0.78,
        0.60,
        0.41,
        0.24,
        0.07,
        -0.10,
        -0.27,
        -0.43,
        -0.59,
        -0.75,
        -0.91,
        -1.08,
        -1.24,
        -1.41,
        -1.56,
        -1.71,
        -1.85,
        -1.98,
        -2.11,
        -2.23,
        -2.35,
        -2.47,
        -2.58,
        -2.69,
        -2.79,
        -2.89,
        -2.99,
        -3.09,
        -3.19,
        -3.28,
        -3.37,
        -3.46,
        -3.55,
        -3.64,
        -3.72,
        -3.80,
        -3.88,
        -3.96,
        -4.04,
        -4.12,
        -4.20,
        -4.27,
        -4.35,
        -4.42,
        -4.49,
        -4.56,
        -4.63,
        -4.70,
        -4.77,
        -4.84,
        -4.90,
        -4.97,
        -5.03,
        -5.10,
        -5.16,
        -5.22,
        -5.28,
        -5.34,
        -5.40,
        -5.46,
        -5.52,
        -5.58,
        -5.63,
        -5.69,
        -5.74,
        -5.80,
        -5.85,
        -5.91,
        -5.96,
        -6.01,
        -6.06,
        -6.11,
        -6.16,
        -6.21,
        -6.26,
        -6.31,
        -6.36,
        -6.41,
        -6.45,
        -6.50,
        -6.55,
        -6.59,
        -6.64,
        -6.68,
        -6.72,
        -6.77,
        -6.81,
        -6.85,
        -6.89,
        -6.93,
        -6.98,
        -7.02,
        -7.06,
        -7.09,
        -7.13,
        -7.17,
        -7.21,
        -7.25,
        -7.28,
        -7.32,
        -7.36,
        -7.39,
        -7.43,
        -7.46,
        -7.50,
        -7.53,
        -7.57,
        -7.60,
        -7.63,
        -7.66,
        -7.70,
        -7.73,
        -7.76,
        -7.79,
        -7.82,
        -7.85,
        -7.88,
        -7.91,
        -7.94,
        -7.97,
        -7.99,
        -8.02,
        -8.05,
        -8.07,
        -8.10,
        -8.13,
        -8.15,
        -8.18,
        -8.20,
        -8.23,
        -8.25,
        -8.28,
        -8.30,
        -8.32,
        -8.34,
        -8.37,
        -8.39,
        -8.41,
        -8.43,
        -8.45,
        -8.47,
        -8.49,
        -8.51,
        -8.53,
        -8.55,
        -8.57,
        -8.59,
        -8.61,
        -8.62,
        -8.64,
        -8.66,
        -8.67,
        -8.69,
        -8.70,
        -8.72,
        -8.74,
        -8.75,
        -8.76,
        -8.78,
        -8.79,
        -8.81,
        -8.82,
        -8.83,
        -8.84,
        -8.86,
        -8.87,
        -8.88,
        -8.89,
        -8.90,
        -8.92,
        -8.93,
        -8.94,
        -8.95,
        -8.96,
        -8.97,
        -8.98,
        -9.00,
        -9.01,
        -9.02,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
        -9.04,
    ]
)
