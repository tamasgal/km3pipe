# coding=utf-8
# cython: profile=True
# Filename: shell.py
# cython: embedsignature=True
# pylint: disable=C0103
"""
Some shell helpers

"""
from __future__ import division, absolute_import, print_function

import os

from .logger import logging

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = logging.getLogger(__name__)  # pylint: disable=C0103


def get_jpp_env(jpp_dir):
    """Return the environment dict of a loaded Jpp env.
    
    The returned env can be passed to `subprocess.Popen("J...", env=env)`
    to execute Jpp commands.

    """
    env = {v[0]:''.join(v[1:]) for v in
           [l.split('=') for l in
            os.popen("source {0}/setenv.sh {0} && env"
                     .format(jpp_dir)).read().split('\n')
            if '=' in l]}
    return env
