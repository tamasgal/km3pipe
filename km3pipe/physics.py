# Filename: physics.py
# pylint: disable=locally-disabled
"""
Cherenkov photon parameters.

"""

import km3io
import numba
import numpy as np
import pandas as pd
import km3pipe as kp

from numba import njit

from .core import Module
from .hardware import Detector
from .dataclasses import Table
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


def cherenkov(calib_hits, track):
    """Compute parameters of Cherenkov photons emitted from a track and hitting a PMT.
    calib_hits is the table of calibrated hits of the track.

    Parameters
    ----------
    calib_hits : kp Table or a DataFrame or a numpy recarray or a dict.
        Table of calibrated hits with the following parameters:
            - pos_x.
            - pos_y.
            - pos_z.
            - dir_x.
            - dir_y.
            - dir_z.
    track : km3io.offline.OfflineBranch or a DataFrame or a numpy recarray or a km3pipe Table or a dict.
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
    ndarray
        a structured array of the physics parameters of Cherenkov photons:
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

    if isinstance(track, km3io.offline.OfflineBranch):
        track_pos = np.array([track.pos_x, track.pos_y, track.pos_z]).T
        track_dir = np.array([track.dir_x, track.dir_y, track.dir_z]).T
        track_t = track.t

    out = _cherenkov(calib_pos, calib_dir, track_pos, track_dir, track_t)

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
    ).reshape(len(out),)


@njit
def _cherenkov(calib_pos, calib_dir, track_pos, track_dir, track_t):
    """calculate Cherenkov photons parameters """
    rows = len(calib_pos)
    out = np.zeros((rows, 8))

    for i in range(rows):
        # (vector PMT) - (vector track position)
        V = calib_pos[i] - track_pos
        L = np.sum(V * track_dir)
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
        out[i][4] = np.sum(out[i][5:8] * calib_dir[i])
    return out


def _get_closest(track_pos, track_dir, meanDU_pos, meanDU_dir):
    """calculate the distance of closest approach """
    # direction track to the mean DU, assumes vertical DU:
    cross = np.cross(track_dir, meanDU_dir)
    dir_to_DU = cross / np.linalg.norm(cross, axis=0)  # normalized vector

    # vector track position to mean DU position, perpendicular to track.
    V = track_pos - meanDU_pos
    d_closest = abs(np.dot(V, dir_to_DU))

    # find z-coordinate of the point of closest approach
    # 1) - vector track position to mean DU position, assuming vertical DU.
    V[2] = 0.0
    V_norm = np.linalg.norm(V, axis=0)

    # 2) distance from the track origin to point of closest approach along the track
    d = np.sqrt(V_norm ** 2 - d_closest ** 2)
    # norm of the track direction along (x,y)
    xy_norm = np.linalg.norm(track_dir[0:2], axis=0)

    # 3) track_pos_Z + distance (along z axis) to closest approach along the track
    z_closest = track_pos[2] + d * (track_dir[2] / xy_norm)

    return d_closest, z_closest


def get_closest(track, du_pos):
    """calculate the distance of closest approach (d_closest) and its coordinate along the z axis (z_closest).
    These calculations aLWAYS assume vertical DU.

    Parameters
    ----------
    track : km3io.offline.OfflineBranch or a DataFrame or a numpy recarray or a km3pipe Table or a dict.
        One track with the following parameters:
            - pos_x.
            - pos_y.
            - pos_z.
            - dir_x.
            - dir_y.
            - dir_z.
    du_pos : a DataFrame or a numpy recarray or a km3pipe Table or a dict.
        du_pos vector with the following information:
            - pos_x.
            - pos_y.
            - pos_z.

    Returns
    -------
    tuple
        (d_closest, z_closest).
    """
    if isinstance(
        du_pos, (dict, pd.core.series.Series, pd.DataFrame, kp.Table, np.ndarray)
    ):
        meanDU_pos = np.array(
            [du_pos["pos_x"], du_pos["pos_y"], du_pos["pos_z"]]
        ).reshape(3,)
        meanDU_dir = np.array([0, 0, 1])  # assumes vertical DU

    if isinstance(
        track, (dict, pd.core.series.Series, pd.DataFrame, kp.Table, np.ndarray)
    ):
        track_pos = np.array([track["pos_x"], track["pos_y"], track["pos_z"]]).reshape(
            3,
        )
        track_dir = np.array([track["dir_x"], track["dir_y"], track["dir_z"]]).reshape(
            3,
        )

    if isinstance(track, km3io.offline.OfflineBranch):
        track_pos = np.array([track.pos_x, track.pos_y, track.pos_z])
        track_dir = np.array([track.dir_x, track.dir_y, track.dir_z])

    return _get_closest(track_pos, track_dir, meanDU_pos, meanDU_dir)
