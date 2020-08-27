# Filename: physics.py
# pylint: disable=locally-disabled
"""
Cherenkov photon parameters.

"""

import math
import numpy as np
import pandas as pd

from .dataclasses import Table
from .logger import get_logger

__author__ = "Zineb ALY"
__copyright__ = "Copyright 2020, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Zineb ALY"
__email__ = "zaly@km3net.de"
__status__ = "Development"

log = get_logger(__name__)

# Physics constants
WATER_INDEX = 1.3499
DN_DL = 0.0298
COS_CHERENKOV = 1 / WATER_INDEX
CHERENKOV_ANGLE_RAD = math.acos(COS_CHERENKOV)
SIN_CHERENKOV = math.sin(CHERENKOV_ANGLE_RAD)
TAN_CHERENKOV = math.tan(CHERENKOV_ANGLE_RAD)
C_LIGHT = 299792458e-9
V_LIGHT_WATER = C_LIGHT / (WATER_INDEX + DN_DL)


def get_cherenkov_photon(calib_hits, track, no_copy=False):
    """Compute parameters of Cherenkov photons emitted from a track and hitting a PMT.
    event is one track (read with km3io, or a DataFrame), and calib_hits is the table of calibrated hits
    of the track.

    Parameters
    ----------
    calib_hits : kp Table of a DataFrame
        Table of calibrated hits.
    track : km3io.offline.OfflineBranch or DataFrame
        One track with the following parameters:
            - pos_x.
            - pos_y.
            - pos_z.
            - dir_x.
            - dir_y.
            - dir_z.
            - t.
    no_copy : bool, optional
        provide a copy of calib_hits.

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

    d = {}

    # (vector PMT) - (vector track position)
    V = np.array(
        [
            calib_hits["pos_x"] - track.pos_x,
            calib_hits["pos_y"] - track.pos_y,
            calib_hits["pos_z"] - track.pos_z,
        ]
    ).T

    # d_photon_closest is the closest distance between the PMT and the track
    # (it is perpendicular to the track)
    L = np.sum(np.array([track.dir_x, track.dir_y, track.dir_z]) * V, axis=1)
    d_2 = np.sum(V * V, axis=1) - np.power(L, 2)  # MUST CHECK IF - OR +

    d_photon_closest = np.sqrt(d_2)

    # distance traveled by the photon from the track to the PMT
    d_photon = d_photon_closest / SIN_CHERENKOV

    # distance along the track where the photon was emitted
    d_track = L - d_photon_closest / TAN_CHERENKOV

    # time of photon travel in [s]
    t_photon = track.t + d_track / C_LIGHT + d_photon / V_LIGHT_WATER

    # photon vector (unitary), which gives photon direction
    track_dir = np.array([track.dir_x, track.dir_y, track.dir_z])
    V_photon = V - (d_track.reshape((-1, 1)) @ track_dir.reshape((1, -1)))
    V_photon = np.divide(
        V_photon, np.linalg.norm(V_photon, axis=1).reshape((-1, 1))
    )  # normalised vector

    # cos angle of impact of photon with respect to the PMT direction
    PMT_dir = calib_hits[["dir_x", "dir_y", "dir_z"]]
    cos_photon_PMT = np.sum(np.array(PMT_dir) * V_photon, axis=1)

    # output
    d["d_photon_closest"] = d_photon_closest
    d["d_photon"] = d_photon
    d["d_track"] = d_track
    d["t_photon"] = t_photon
    d["cos_photon_PMT"] = cos_photon_PMT
    d["dir_x_photon"] = V_photon[:, 0]
    d["dir_y_photon"] = V_photon[:, 1]
    d["dir_z_photon"] = V_photon[:, 2]

    return pd.DataFrame(d)
