# Filename: test_io.py
# pylint: disable=locally-disabled,C0111,R0904,C0301,C0103,W0212
import tempfile

import tables

from km3pipe.tools import istype
from km3pipe.testing import TestCase, patch, Mock
from km3pipe.io import GenericPump, read_calibration

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

    def test_filenames_should_be_valid_iterable(self):
        with self.assertRaises(TypeError):
            GenericPump(None)

    def test_init_raises_ioerror_for_mixed_filetypes(self):
        with self.assertRaises(IOError):
            GenericPump(['a.a', 'b.b'])

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
        with self.assertRaises(SystemExit):
            GenericPump(fnames)


class TestReadCalibration(TestCase):
    def test_call(self):
        read_calibration()

    @patch('km3pipe.calib.Calibration')
    def test_call_with_detx(self, mock_calibration):
        read_calibration(detx='a')
        mock_calibration.assert_called_with(filename='a')

    @patch('km3pipe.calib.Calibration')
    def test_call_with_det_id(self, mock_calibration):
        det_id = 1
        read_calibration(det_id=det_id)
        mock_calibration.assert_called_with(det_id=det_id)

    @patch('km3pipe.calib.Calibration')
    def test_call_with_negative_det_id(self, mock_calibration):
        det_id = -1
        read_calibration(det_id=det_id)
        mock_calibration.assert_not_called()
