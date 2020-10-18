# Filename: test_cmd.py
# pylint: disable=C0111,E1003,R0904,C0103,R0201,C0102

from km3pipe.testing import TestCase, patch
from km3pipe.cmd import update_km3pipe

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

KM3PIPE_GIT = "http://git.km3net.de/km3py/km3pipe.git"


class TestUpdateKm3pipe(TestCase):
    @patch("km3pipe.cmd.os")
    def test_update_without_args_updates_master(self, mock_os):
        update_km3pipe()
        expected = "pip install -U git+{0}@master".format(KM3PIPE_GIT)
        mock_os.system.assert_called_with(expected)
        update_km3pipe("")
        expected = "pip install -U git+{0}@master".format(KM3PIPE_GIT)
        mock_os.system.assert_called_with(expected)
        update_km3pipe(None)
        expected = "pip install -U git+{0}@master".format(KM3PIPE_GIT)
        mock_os.system.assert_called_with(expected)

    @patch("km3pipe.cmd.os")
    def test_update_branch(self, mock_os):
        branch = "foo"
        update_km3pipe(branch)
        expected = "pip install -U git+{0}@{1}".format(KM3PIPE_GIT, branch)
        mock_os.system.assert_called_with(expected)
