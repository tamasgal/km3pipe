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

from .__version__ import version, VERSION_INFO  # noqa

if not __KM3PIPE_SETUP__:
    from .core import (Pipeline, Module, Pump, Blob, Run, Geometry)  # noqa
    import km3pipe.core  # noqa
    import km3pipe.dataclasses  # noqa
    # import km3pipe.calib  # noqa
    # import km3pipe.cmd  # noqa
    # import km3pipe.common  # noqa
    # import km3pipe.config  # noqa
    # import km3pipe.constants  # noqa
    # import km3pipe.controlhost  # noqa
    # import km3pipe.db  # noqa
    # import km3pipe.hardware  # noqa
    # import km3pipe.io  # noqa
    # import km3pipe.logger  # noqa
    # import km3pipe.math  # noqa
    # import km3pipe.mc  # noqa
    # import km3pipe.shell  # noqa
    # # import km3pipe.srv  # noqa
    # # import km3pipe.style  # noqa
    # import km3pipe.sys  # noqa
    # # import km3pipe.testing  # noqa
    # import km3pipe.time  # noqa
    # import km3pipe.tools  # noqa


__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Thomas Heid"]
__license__ = "MIT"
__version__ = version
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"
