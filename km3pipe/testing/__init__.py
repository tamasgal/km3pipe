# Filename: __init__.py
"""
Common unit testing support for km3pipe.

"""

from io import BytesIO, StringIO  # noqa

from unittest import TestCase  # noqa
from mock import MagicMock  # noqa
from mock import Mock  # noqa
from mock import patch  # noqa

from numpy.testing import assert_allclose  # noqa
import pytest  # noqa

skip = pytest.mark.skip
skipif = pytest.mark.skipif

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"
