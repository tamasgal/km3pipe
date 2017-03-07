# coding=utf-8
# cython: profile=True
# Filename: tools.pyx
# cython: embedsignature=True
# pylint: disable=C0103
"""
Some unsorted, frequently used logic.

"""
from __future__ import division, absolute_import, print_function

import os
import base64
import subprocess


from .logger import logging

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = logging.getLogger(__name__)  # pylint: disable=C0103


def ifiles(irods_path):
    """Return a list of filenames for given iRODS path (recursively)"""
    raw_output = subprocess.check_output("ils -r --bundle {0}"
                                         "    | grep 'Bundle file:'"
                                         "    | awk '{{print $3}}'"
                                         .format(irods_path), shell=True)
    filenames = raw_output.strip().split("\n")
    return filenames


def token_urlsafe(nbytes=32):
    """Return a random URL-safe text string, in Base64 encoding.

    This is taken and slightly modified from the Python 3.6 stdlib.

    The string has *nbytes* random bytes.  If *nbytes* is ``None``
    or not supplied, a reasonable default is used.

    >>> token_urlsafe(16)  #doctest:+SKIP
    'Drmhze6EPcv0fN_81Bj-nA'

    """
    tok = os.urandom(nbytes)
    return base64.urlsafe_b64encode(tok).rstrip(b'=').decode('ascii')


def insert_prefix_to_dtype(arr, prefix):
    new_cols = [prefix + '_' + col for col in arr.dtype.names]
    arr.dtype.names = new_cols
    return arr


def prettyln(text, fill='-', align='^', prefix='[ ', suffix=' ]', length=69):
    """Wrap `text` in a pretty line with maximum length."""
    text = '{prefix}{0}{suffix}'.format(text, prefix=prefix, suffix=suffix)
    print("{0:{fill}{align}{length}}"
          .format(text, fill=fill, align=align, length=length))


def irods_filepath(det_id, run_id):
    """Generate the iRODS filepath for given detector (O)ID and run ID"""
    data_path = "/in2p3/km3net/data/raw/sea"
    from km3pipe.db import DBManager
    if not isinstance(det_id, int):
        dts = DBManager().detectors
        det_id = int(dts[dts.OID == det_id].SERIALNUMBER.values[0])
    return data_path + "/KM3NeT_{0:08}/{2}/KM3NeT_{0:08}_{1:08}.root" \
           .format(det_id, run_id, run_id//1000)
