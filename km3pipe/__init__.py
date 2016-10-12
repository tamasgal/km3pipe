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
    import km3pipe
    from km3pipe.core import (Pipeline, Module, Pump, Blob, Run,  # noqa
                              Geometry, AanetGeometry)
    from km3pipe import io  # noqa
    from km3pipe import utils  # noqa
    from km3pipe import srv  # noqa
    from km3pipe.srv import srv_event  # noqa
    from km3pipe.io import GenericPump, read_hdf5  # noqa

    import os

    mplstyle = os.path.dirname(km3pipe.__file__) + '/kp-data/km3pipe.mplstyle'

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Thomas Heid"]
__license__ = "MIT"
__version__ = version
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"
