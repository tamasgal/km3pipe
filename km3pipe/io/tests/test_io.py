# Filename: test_io.py
# pylint: disable=locally-disabled,C0111,R0904,C0301,C0103,W0212
import tempfile

import tables

from km3pipe.tools import istype
from km3pipe.testing import TestCase, patch, Mock
from km3pipe.io import GenericPump

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2018, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestGenericPump(TestCase):
    def test_init_raies_filenotfounderror_for_nonexistinent_file(self):
        with self.assertRaises(ValueError):
            GenericPump("unsupport-file")

    def test_init_evt_with_one_file(self):
        fobj = tempfile.NamedTemporaryFile(delete=True, suffix='.evt')
        fname = str(fobj.name)
        p = GenericPump(fname)
        fobj.close()
        assert istype(p, 'EvtPump')
        assert fname == p.filename

    def test_init_h5_with_one_file(self):
        fobj = tempfile.NamedTemporaryFile(delete=True, suffix='.h5')
        fname = str(fobj.name)
        with self.assertRaises(tables.HDF5ExtError):
            p = GenericPump(fname)
        fobj.close()
