# Filename: calib.py
# pylint: disable=locally-disabled
"""
Calibration.

"""

import numpy as np
from pandas import DataFrame

from .core import Module
from .hardware import Detector
from .dataclasses import Table
from .dataclass_templates import TEMPLATES
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
except (ImportError, OSError):
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
    __name__ = 'Calibration'
    name = 'Calibration'

    def configure(self):
        self._should_apply = self.get('apply', default=False)
        self.filename = self.get('filename')
        self.det_id = self.get('det_id')
        self.t0set = self.get('t0set')
        self.calibration = self.get('calibration')
        self.detector = self.get('detector')
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
            apply_t0_nb(hits.time, hits.dom_id, hits.channel_id,
                        self._lookup_tables)
        else:
            n = len(hits)
            cal = np.empty(n)
            lookup = self._calib_by_dom_and_channel
            for i in range(n):
                calib = lookup[hits['dom_id']
                               [i]][hits['channel_id'][i]]
                cal[i] = calib[6]
            hits.time += cal
        return hits

    def apply(self, hits):
        """Add x, y, z, t0 (and du, floor if DataFrame) columns to the hits.

        """
        if isinstance(hits, DataFrame):
            # do we ever see McHits here?
            hits = Table.from_template(hits, 'Hits')
        if hasattr(hits, 'dom_id') and hasattr(hits, 'channel_id'):
            return self._apply_to_hits(hits)
        elif hasattr(hits, 'pmt_id'):
            return self._apply_to_mchits(hits)
        else:
            raise TypeError("Don't know how to apply calibration to '{0}'. "
                            "We need at least 'dom_id' and 'channel_id', or "
                            "'pmt_id'."
                            .format(hits.name))

    def _apply_to_hits(self, hits):
        """Append the position, direction and t0 columns and add t0 to time"""
        n = len(hits)
        cal = np.empty((n, 9))
        lookup = self._calib_by_dom_and_channel
        for i in range(n):
            calib = lookup[hits['dom_id'][i]][hits['channel_id'][i]]
            cal[i] = calib
        h = np.empty(n, TEMPLATES['CalibHits']['dtype'])
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
        h['group_id'] = hits['group_id']
        return Table.from_template(h, 'CalibHits')

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

        return hits.append_columns(
            ['dir_x', 'dir_y', 'dir_z', 'du', 'floor',
             'pos_x', 'pos_y', 'pos_z', 't0'],
            [dir_x, dir_y, dir_z, du, floor, pos_x, pos_y, pos_z, t0]
        )

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
