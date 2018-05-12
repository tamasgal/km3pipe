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
        # We expect just an HDF5ExtError if "everything" goes well
        with self.assertRaises(tables.HDF5ExtError):
            GenericPump(fname)
        fobj.close()

    def test_init_h5_with_multiple_files(self):
        fobj1 = tempfile.NamedTemporaryFile(delete=True, suffix='.h5')
        fobj2 = tempfile.NamedTemporaryFile(delete=True, suffix='.h5')
        fname1 = str(fobj1.name)
        fname2 = str(fobj2.name)
        # We expect just an HDF5ExtError if "everything" goes well
        with self.assertRaises(tables.HDF5ExtError):
            GenericPump([fname1, fname2])
        fobj1.close()
        fobj2.close()

    def test_init_h5_with_multiple_files_where_one_nonexistent(self):
        fobj1 = tempfile.NamedTemporaryFile(delete=True, suffix='.h5')
        fobj2 = tempfile.NamedTemporaryFile(delete=True, suffix='.h5')
        fname1 = str(fobj1.name)
        fname2 = str(fobj2.name)
        fname3 = "nonexistent-file.h5"
        # We expect just an HDF5ExtError if "everything" goes well
        with self.assertRaises(tables.HDF5ExtError):
            GenericPump([fname1, fname2, fname3])
        fobj1.close()
        fobj2.close()

    def test_init_h5_with_nonexistent_files(self):
        fnames = ["nonexistent-file{}.h5".format(i) for i in range(3)]
        with self.assertRaises(FileNotFoundError):
            GenericPump(fnames)
