# coding=utf-8
# Filename: ahrs.py
# pylint: disable=locally-disabled
"""
AHRS calibration.

"""
from __future__ import division, absolute_import, print_function

import io
from functools import lru_cache
import xml.etree.ElementTree as ET

import numpy as np
from numpy import cos, sin, arctan2
import km3pipe as kp

__author__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = kp.logger.logging.getLogger(__name__)  # pylint: disable=C0103
# log.setLevel(logging.DEBUG)


def fit_ahrs(A, H, Aoff, Arot, Hoff, Hrot):
    """Calculate yaw, pitch and roll for given A/H and calibration set.

    Author: Vladimir Kulikovsky

    Parameters
    ----------
    A: list, tuple or numpy.array of shape (3,)
    H: list, tuple or numpy.array of shape (3,)
    Aoff: numpy.array of shape(3,)
    Arot: numpy.array of shape(3, 3)
    Hoff: numpy.array of shape(3,)
    Hrot: numpy.array of shape(3, 3)

    Returns
    -------
    yaw, pitch, roll

    """
    Acal = np.dot(A-Aoff, Arot)
    Hcal = np.dot(H-Hoff, Hrot)

    #invert axis for DOM upside down
    for i in (1, 2):
        Acal[i] = -Acal[i]
        Hcal[i] = -Hcal[i]

    roll = arctan2(-Acal[1], -Acal[2])
    pitch= arctan2(Acal[0], np.sqrt(Acal[1]*Acal[1] + Acal[2]*Acal[2]))
    yaw = arctan2(Hcal[2]*sin(roll) - Hcal[1]*cos(roll),
                  sum(Hcal[0]*cos(pitch),
                      Hcal[1]*sin(pitch)*sin(roll),
                      Hcal[2]*sin(pitch)*cos(roll)))

    #yaw = (yaw + magnetic_declination + 360 ) % 360
    yaw = np.degrees(yaw)
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    return yaw, pitch, roll


@lru_cache(maxsize=None, typed=False)
def get_latest_ahrs_calibration(clb_upi, max_version=3, db=None):
    """Retrieve the latest AHRS calibration data for a given CLB

    Parameters
    ----------
    clb_upi: str
    max_version: int, maximum version to check, optional
    db: DBManager(), optional

    Returns
    -------
    Aoff: numpy.array with shape(3,)
    Arot: numpy.array with shape(3,3)
    Hoff: numpy.array with shape(3,)
    Hrot: numpy.array with shape(3,3)

    or None if no calibration found.

    """
    ahrs_upi = kp.db.clbupi2ahrsupi(clb_upi)

    if db is None:
        db = kp.db.DBManager()

    for version in range(max_version, 0, -1):
        raw_data = db._get_content("show_product_test.htm?upi={0}&"
                                   "testtype=AHRS-CALIBRATION-v{1}&n=1&out=xml"
                                   .format(ahrs_upi, version)) \
                                   .replace('\n', '')
        try:
            xroot = ET.parse(io.StringIO(raw_data)).getroot()
        except ET.ParseError:
            continue
        else:
            return _extract_calibration(xroot)

    return None


def _extract_calibration(xroot):
    """Extract AHRS calibration information from XML root.

    Parameters
    ----------
    xroot: XML root


    Returns
    -------
    Aoff: numpy.array with shape(3,)
    Arot: numpy.array with shape(3,3)
    Hoff: numpy.array with shape(3,)
    Hrot: numpy.array with shape(3,3)

    """
    names = [c.text for c in xroot.findall(".//Name")]
    val = [[i.text for i in c] for c in xroot.findall(".//Values")]

    # The fields has to be reindeced, these are the index mappings
    col_ic = [int(v) for v in val[names.index("AHRS_Matrix_Column(-)")]]
    try:
        row_ic = [int(v) for v in val[names.index("AHRS_Matrix_Row(-)")]]
    except ValueError:
        row_ic = [2, 2, 2, 1, 1, 1, 0, 0, 0]
    try:
        vec_ic = [int(v) for v in val[names.index("AHRS_Vector_Index(-)")]]
    except ValueError:
        vec_ic = [2, 1, 0]

    Aoff_ix = names.index("AHRS_Acceleration_Offset(g/ms^2-)")
    Arot_ix = names.index("AHRS_Acceleration_Rotation(-)")
    Hrot_ix = names.index("AHRS_Magnetic_Rotation(-)")

    Arot = np.array(val[Arot_ix]).reshape(3, 3)[col_ic, row_ic].reshape(3, 3)
    Aoff = np.array(val[Aoff_ix])[vec_ic]
    Hrot = np.array(val[Hrot_ix]).reshape(3, 3)[col_ic, row_ic].reshape(3, 3)

    Hoff = []
    for q in 'XYZ':
        values = []
        for t in ('Min', 'Max'):
            ix = names.index("AHRS_Magnetic_{}{}(G-)".format(q, t))
            values.append(float(val[ix][0]))
        Hoff.append(sum(values) / 2.)
    Hoff = np.array(Hoff)

    return Aoff, Arot, Hoff, Hrot
