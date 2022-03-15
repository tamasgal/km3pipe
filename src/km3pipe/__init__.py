# Filename: __init__.py
"""
The extemporary KM3NeT analysis framework.

"""
from pkg_resources import get_distribution, DistributionNotFound

try:
    version = get_distribution(__name__).version
except DistributionNotFound:
    version = "unknown version"

try:
    __KM3PIPE_SETUP__
except NameError:
    __KM3PIPE_SETUP__ = False

if not __KM3PIPE_SETUP__:
    from . import logger  # noqa
    from .dataclasses import Table, NDArray  # noqa
    from . import dataclasses  # noqa
    from . import calib  # noqa
    from . import cmd  # noqa
    from . import constants  # noqa
    from . import controlhost  # noqa
    from . import hardware  # noqa
    from . import io  # noqa
    from . import math  # noqa
    from . import mc  # noqa
    from . import physics
    from . import shell  # noqa
    from . import style  # noqa
    from . import sys  # noqa

    # from . import testing     # noqa
    from . import time  # noqa
    from . import tools  # noqa

    from thepipe import (
        Pipeline,
        Module,
        Blob,
        Provenance,
    )  # reexport the provenance handler

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Thomas Heid"]
__license__ = "MIT"
__version__ = version
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"
