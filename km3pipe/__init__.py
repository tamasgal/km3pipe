# coding=utf-8
# Filename: __init__.py
"""
The extemporary KM3NeT analysis framework.

"""
from __future__ import division, absolute_import, print_function


from km3pipe.__version__ import version, version_info  # noqa

#try:
import pyximport
pyximport.install()

from km3pipe.core import (Pipeline, Module, Pump, Blob,  # noqa
                          Geometry, AanetGeometry)
#    from km3pipe import pumps  # noqa
#except ImportError:
#    print("Some modules could not be imported. Ignore this if you're "
#          "installing or upgrading KM3Pipe.")

__author__ = "Tamas Gal"
__copyright__ = ("Copyright 2015, Tamas Gal and the KM3NeT collaboration "
                 "(http://km3net.org)")
__credits__ = []
__license__ = "MIT"
__version__ = version
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"
