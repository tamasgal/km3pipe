# Filename: db.py
# pylint: disable=locally-disabled
"""
Database utilities.

"""
from km3db import DBManager, StreamDS, CLBMap
from km3db.tools import clbupi2compassupi, show_compass_calibration


def clbupi2ahrsupi(clb_upi):
    """Return UPI from CLB UPI. Wrap clbupi2compassupi for back-compatibility."""
    log.deprecation("clbupi2ahrsupi is deprecated ! You should use clbupi2compassupi.")
    upi = clbupi2compassupi(clb_upi)
    if upi.split("/")[1] != "AHRS":
        log.warning("clbupi2ahrsupi() is returning a LSM303 UPI : {}".format(upi))
    return upi


def show_ahrs_calibration(clb_upi, version="3"):
    """Show AHRS calibration data for given `clb_upi`."""
    log.deprecation(
        "show_ahrs_calibration is deprecated ! You should use show_compass_calibration()."
    )
    show_compass_calibration(clb_upi, version=version)
