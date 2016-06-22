# coding=utf-8
# Filename: __init__.py
"""
The extemporary KM3NeT analysis framework.

"""
from __future__ import division, absolute_import, print_function


try:
    __KM3PIPE_SETUP__
except NameError:
    __KM3PIPE_SETUP__ = False

from km3pipe.__version__ import version, version_info  # noqa

if not __KM3PIPE_SETUP__:
    from km3pipe.core import (Pipeline, Module, Pump, Blob,  # noqa
                              Geometry, AanetGeometry)
    from km3pipe import pumps  # noqa
    from km3pipe import utils  # noqa
    from km3pipe import srv  # noqa
    from km3pipe.srv import srv_event  # noqa
    from km3pipe.tools import read_hdf5, open_hdf5, read_reco  # noqa
    from km3pipe.pumps import GenericPump

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Thomas Heid"]
__license__ = "MIT"
__version__ = version
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"
