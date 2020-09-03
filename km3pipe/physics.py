# Filename: physics.py
# pylint: disable=locally-disabled
"""
Cherenkov photon parameters.

"""

import numba
import numpy as np
import pandas as pd
import km3pipe as kp

from numba import njit

from .core import Module
from .dataclasses import Table
from .tools import istype
from .logger import get_logger
from .constants import SIN_CHERENKOV, TAN_CHERENKOV, V_LIGHT_WATER, C_LIGHT

__author__ = "Zineb ALY"
__copyright__ = "Copyright 2020, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Zineb ALY"
__email__ = "zaly@km3net.de"
__status__ = "Development"

log = get_logger(__name__)

try:
    import numba as nb
except (ImportError, OSError):
    HAVE_NUMBA = False
    jit = lambda f: f
    log.warning("No numba detected, consider `pip install numba` for more speed!")
else:
    try:
        from numba.typed import Dict
    except ImportError:
        log.warning("Please update numba (0.43+) to have dictionary support!")
        HAVE_NUMBA = False
        jit = lambda f: f
    else:
        HAVE_NUMBA = True
        from numba import jit


class CherekovPhotons(Module):
    """A Module to access Cherenkov Photons parameters."""

    __name__ = "CherekovPhotons"
    name = "CherekovPhotons"

    def configure(self):
        self._should_apply = self.get("apply", default=True)

    def process(self, blob, keys=["CalibHits", "track"], outkey="CherenkovPhotons"):
        if self._should_apply:
            blob[outkey] = self.apply(blob[keys[0]], blob[keys[1]])
        return blob

    def apply(self, calib_hits, track):
        return get_cherenkov(calib_hits, track)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "CherekovPhotons"


@njit
def _get_cherenkov(calib_pos, calib_dir, track_pos, track_dir, track_t):
    """calculate Cherenkov photons parameters """
    rows = len(calib_pos)
    out = np.zeros((rows, 8))

    for i in range(rows):
        # (vector PMT) - (vector track position)
        V = calib_pos[i] - track_pos
        L = np.sum(V * track_pos)
        out[i][0] = np.sqrt(np.sum(V * V) - L * L)  # d_photon_closest
        out[i][1] = out[i][0] / SIN_CHERENKOV  # d_photon
        out[i][2] = L - out[i][0] / TAN_CHERENKOV  # d_track
        out[i][3] = (
            track_t + out[i][2] / C_LIGHT + out[i][1] / V_LIGHT_WATER
        )  # t_photon
        V_photon = V - (out[i][2] * track_dir)  # photon position
        norm = np.sqrt(np.sum(V_photon * V_photon))
        out[i][5:8] = V_photon / norm  # photon direction

        # cos angle of impact of photon with respect to the PMT direction
        out[i][4] = np.sum(out[i][4:7] * calib_dir[i])
    return out


def get_cherenkov(calib_hits, track):
    """Compute parameters of Cherenkov photons emitted from a track and hitting a PMT.
    calib_hits is the table of calibrated hits of the track.

    Parameters
    ----------
    calib_hits : kp Table or a DataFrame or a numpy recarray.
        Table of calibrated hits with the following parameters:
            - pos_x.
            - pos_y.
            - pos_z.
            - dir_x.
            - dir_y.
            - dir_z.
            - time.
    track : km3io.offline.OfflineBranch or a DataFrame or a numpy recarray.
        One track with the following parameters:
            - pos_x.
            - pos_y.
            - pos_z.
            - dir_x.
            - dir_y.
            - dir_z.
            - t.

    Returns
    -------
    Dataframe
        a table of the physics parameters of Cherenkov photons:
            - d_photon_closest: the closest distance between the PMT and the track
            (it is perpendicular to the track).
            - d_photon: distance traveled by the photon from the track to the PMT.
            - d_track: distance along the track where the photon was emitted.
            - t_photon: time of photon travel in [s].
            - cos_photon_PMT: cos angle of impact of photon with respect to the PMT direction:
            - dir_x_photon, dir_y_photon, dir_z_photon: photon directions.
    """

    if isinstance(calib_hits, (dict, pd.DataFrame, np.ndarray, kp.Table)):
        calib_pos = np.array(
            [calib_hits["pos_x"], calib_hits["pos_y"], calib_hits["pos_z"]]
        ).T
        calib_dir = np.array(
            [calib_hits["dir_x"], calib_hits["dir_y"], calib_hits["dir_z"]]
        ).T

    if isinstance(track, (dict, pd.core.series.Series, pd.DataFrame)):
        track_pos = np.array([track["pos_x"], track["pos_y"], track["pos_z"]]).T
        track_dir = np.array([track["dir_x"], track["dir_y"], track["dir_z"]]).T
        track_t = track["t"]

    if isinstance(track, (kp.Table, np.ndarray)):
        track_pos = np.array([track["pos_x"], track["pos_y"], track["pos_z"]]).reshape(
            3,
        )
        track_dir = np.array([track["dir_x"], track["dir_y"], track["dir_z"]]).reshape(
            3,
        )
        track_t = track["t"][0]

    out = _get_cherenkov(calib_pos, calib_dir, track_pos, track_dir, track_t)

    return out.view(
        dtype=[
            ("d_photon_closest", "<f8"),
            ("d_photon", "<f8"),
            ("d_track", "<f8"),
            ("t_photon", "<f8"),
            ("cos_photon_PMT", "<f8"),
            ("dir_x_photon", "<f8"),
            ("dir_y_photon", "<f8"),
            ("dir_z_photon", "<f8"),
        ]
    ).reshape(
        len(out),
    )
