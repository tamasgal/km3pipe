# Filename: common.py
# pylint: disable=locally-disabled
"""
Commonly used imports.

"""

try:
    from cStringIO import StringIO
except ImportError:
    try:
        from StringIO import StringIO
    except ImportError:
        from io import StringIO  # noqa

try:
    from Queue import Queue, Empty  # noqa
except ImportError:
    from queue import Queue, Empty  # noqa


__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"
