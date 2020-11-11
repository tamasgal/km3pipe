# Filename: __init__.py
"""
A collection of io for different kinds of data formats.

"""

import os.path

import numpy as np

from .evt import EvtPump  # noqa
from .clb import CLBPump  # noqa
from .ch import CHPump  # noqa
from .hdf5 import HDF5Pump, HDF5Sink, HDF5MetaData  # noqa
from .offline import OfflinePump
from . import online
from . import offline
from . import daq

from km3pipe.logger import get_logger

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal, Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"
