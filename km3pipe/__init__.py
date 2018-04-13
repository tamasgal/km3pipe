# coding=utf-8
# Filename: __init__.py
"""
The extemporary KM3NeT analysis framework.

"""
from __future__ import division, absolute_import, print_function

from .__version__ import version, VERSION_INFO  # noqa

try:
    __KM3PIPE_SETUP__
except NameError:
    __KM3PIPE_SETUP__ = False

if not __KM3PIPE_SETUP__:
    from .core import (Pipeline, Module, Pump, Blob, Run, Geometry)  # noqa
    from . import core
    from . import dataclasses
    # from . import calib
    from . import cmd
    from . import common
    from . import config
    from . import constants
    from . import controlhost
    from . import db
    from . import hardware
    from . import io
    from . import logger
    from . import math
    from . import mc
    from . import shell
    # from . import srv
    from . import style
    from . import sys
    # from . import testing
    from . import time
    from . import tools


__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Thomas Heid"]
__license__ = "MIT"
__version__ = version
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"
